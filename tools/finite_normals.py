#!/usr/bin/env python3
"""Replaces non-finite NORMAL values in a GLB in place, pointing them up, and rewrites the accessors' bounds. finite_normals.py model.glb..."""
import json, struct, sys
import numpy as np
for path in sys.argv[1:]:
    data = bytearray(open(path, 'rb').read()); length = struct.unpack_from('<I', data, 12)[0]; tree = json.loads(data[20:20 + length].decode().replace('NaN', 'null'))
    binary = 20 + length + 8; fixed = 0
    for mesh in tree['meshes']:
        for primitive in mesh['primitives']:
            if 'NORMAL' not in primitive['attributes']: continue
            accessor = tree['accessors'][primitive['attributes']['NORMAL']]; view = tree['bufferViews'][accessor['bufferView']]
            start = binary + view.get('byteOffset', 0) + accessor.get('byteOffset', 0); stride = view.get('byteStride', 12)
            values = np.ndarray((accessor['count'], 3), np.float32, buffer=data, offset=start, strides=(stride, 4))
            broken = ~np.isfinite(values).all(1); values[broken] = (0, 1, 0); fixed += int(broken.sum())
            accessor['min'], accessor['max'] = values.min(0).tolist(), values.max(0).tolist()
    if not fixed: continue  # Rewriting an untouched file would change its bytes, and the paint stage hashes them.
    blob = json.dumps(tree, separators=(',', ':')).encode(); blob += b' ' * (-len(blob) % 4); rest = bytes(data[20 + length:])
    open(path, 'wb').write(struct.pack('<III', 0x46546C67, 2, 20 + len(blob) + len(rest)) + struct.pack('<II', len(blob), 0x4E4F534A) + blob + rest)
    print('FINITE_NORMALS', path, fixed, flush=True)
