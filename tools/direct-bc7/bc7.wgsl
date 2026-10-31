// Copyright (c) 2025, Nils Hasenbanck
// Copyright (c) 2016-2024, Intel Corporation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of
// the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

struct Uniforms {
    width: u32,
    height: u32,
    texture_y_offset: u32,
    blocks_offset: u32,
}

struct Settings {
    refine_iterations: array<u32, 8>,
    mode_selection: array<u32, 4>,
    skip_mode2: u32,
    fast_skip_threshold_mode1: u32,
    fast_skip_threshold_mode3: u32,
    fast_skip_threshold_mode7: u32,
    mode45_channel0: u32,
    refine_iterations_channel: u32,
    channels: u32,
}

struct State {
    data: array<u32, 5>,
    best_err: f32,
    opaque_err: f32,
}

struct Mode45Parameters {
    qep: array<i32, 8>,
    qblock: vec2<u32>,
    aqep: vec2<i32>,
    aqblock: vec2<u32>,
    rotation: u32,
    swap: u32,
}

// Packed read-only tables are shared across invocations, rather than copied
// into dynamically indexed function-local arrays in every encoder invocation.
@group(0) @binding(0) var<uniform> encoder_tables:array<vec4<u32>,96>;
fn encoder_table(index:u32)->u32 {return encoder_tables[index/4u][index%4u];}
@group(0) @binding(0) var source_texture: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> block_buffer: array<u32>;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;
@group(0) @binding(3) var<storage, read> settings: Settings;

fn sq(x: f32) -> f32 {
    return x * x;
}

fn rsqrt(x: f32) -> f32 {
    return 1.0 / sqrt(x);
}

fn load_block_interleaved_rgba(block: ptr<function, array<f32, 64>>, xx: u32, yy: u32) {
    for (var y = 0u; y < 4u; y++) {
        for (var x = 0u; x < 4u; x++) {
            let pixel_x = xx * 4u + x;
            let pixel_y = yy * 4u + y + uniforms.texture_y_offset;
            let rgba = textureLoad(source_texture, vec2<u32>(pixel_x, pixel_y), 0);

            (*block)[16u * 0u + y * 4u + x] = rgba.r * 255.0;
            (*block)[16u * 1u + y * 4u + x] = rgba.g * 255.0;
            (*block)[16u * 2u + y * 4u + x] = rgba.b * 255.0;
            (*block)[16u * 3u + y * 4u + x] = rgba.a * 255.0;
        }
    }
}

fn store_data(state: ptr<function, State>, block_width: u32, xx: u32, yy: u32) {
    let offset = uniforms.blocks_offset + (yy * block_width * 4u + xx * 4u);

    block_buffer[offset + 0] = (*state).data[0];
    block_buffer[offset + 1] = (*state).data[1];
    block_buffer[offset + 2] = (*state).data[2];
    block_buffer[offset + 3] = (*state).data[3];
}

fn get_unquant_value(bits: u32, index: i32) -> i32 {
    switch (bits) {
        case 2u: {
            const table = array<i32, 16>(0, 21, 43, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
            return table[index];
        }
        case 3u: {
            const table = array<i32, 16>(0, 9, 18, 27, 37, 46, 55, 64, 0, 0, 0, 0, 0, 0, 0, 0);
            return table[index];
        }
        default: {
            const table = array<i32, 16>(0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64);
            return table[index];
        }
    }
}

fn get_pattern(part_id: i32) -> u32 {

    return encoder_table(0u+u32(part_id));
}

fn get_pattern_mask(part_id: i32, j: u32) -> u32 {

    let mask_packed = encoder_table(128u+u32(part_id));
    let mask0 = mask_packed & 0xFFFFu;
    let mask1 = mask_packed >> 16u;

    return select(select(mask1, mask0, j == 0), ~mask0 & ~mask1, j == 2);
}

fn get_skips(part_id: i32) -> vec3<u32> {

    let skip_packed = encoder_table(256u+u32(part_id));

    return vec3<u32>(0u, skip_packed >> 4u, skip_packed & 15u);
}

// Principal Component Analysis (PCA) bound
fn get_pca_bound(covar: array<f32, 10>, channels: u32) -> f32 {
    const power_iterations = 4u; // Quite approximative, but enough for bounding

    var covar_scaled = covar;
    let inv_var = 1.0 / (256.0 * 256.0);
    for (var k = 0u; k < 10u; k++) {
        covar_scaled[k] *= inv_var;
    }

    let eps = sq(0.001);
    covar_scaled[0] += eps;
    covar_scaled[4] += eps;
    covar_scaled[7] += eps;

    var axis: vec4<f32>;
    compute_axis(&axis, &covar_scaled, power_iterations, channels);

    var a_vec: vec4<f32>;
    if (channels == 3u) {
        ssymv3(&a_vec, &covar_scaled, &axis);
    } else if (channels == 4u) {
        ssymv4(&a_vec, &covar_scaled, &axis);
    }

    var sq_sum = 0.0;
    for (var p = 0u; p < channels; p++) {
        sq_sum += sq(a_vec[p]);
    }
    let lambda = sqrt(sq_sum);

    var bound = covar_scaled[0] + covar_scaled[4] + covar_scaled[7];
    if (channels == 4u) {
        bound += covar_scaled[9];
    }
    bound -= lambda;
    bound = max(bound, 0.0);

    return bound;
}

fn ssymv3(a: ptr<function, vec4<f32>>, covar: ptr<function, array<f32, 10>>, b: ptr<function, vec4<f32>>) {
    (*a)[0] = (*covar)[0] * (*b)[0] + (*covar)[1] * (*b)[1] + (*covar)[2] * (*b)[2];
    (*a)[1] = (*covar)[1] * (*b)[0] + (*covar)[4] * (*b)[1] + (*covar)[5] * (*b)[2];
    (*a)[2] = (*covar)[2] * (*b)[0] + (*covar)[5] * (*b)[1] + (*covar)[7] * (*b)[2];
}

fn ssymv4(a: ptr<function, vec4<f32>>, covar: ptr<function, array<f32, 10>>, b: ptr<function, vec4<f32>>) {
    (*a)[0] = (*covar)[0] * (*b)[0] + (*covar)[1] * (*b)[1] + (*covar)[2] * (*b)[2] + (*covar)[3] * (*b)[3];
    (*a)[1] = (*covar)[1] * (*b)[0] + (*covar)[4] * (*b)[1] + (*covar)[5] * (*b)[2] + (*covar)[6] * (*b)[3];
    (*a)[2] = (*covar)[2] * (*b)[0] + (*covar)[5] * (*b)[1] + (*covar)[7] * (*b)[2] + (*covar)[8] * (*b)[3];
    (*a)[3] = (*covar)[3] * (*b)[0] + (*covar)[6] * (*b)[1] + (*covar)[8] * (*b)[2] + (*covar)[9] * (*b)[3];
}

fn compute_axis(axis: ptr<function, vec4<f32>>, covar: ptr<function, array<f32, 10>>, power_iterations: u32, channels: u32) {
    var a_vec = vec4<f32>(1.0, 1.0, 1.0, 1.0);

    for (var i = 0u; i < power_iterations; i++) {
        if (channels == 3u) {
            ssymv3(axis, covar, &a_vec);
        } else if (channels == 4u) {
            ssymv4(axis, covar, &a_vec);
        }

        for (var p = 0u; p < channels; p++) {
            a_vec[p] = (*axis)[p];
        }

        // Renormalize every other iteration
        if (i % 2u == 1u) {
            var norm_sq = 0.0;
            for (var p = 0u; p < channels; p++) {
                norm_sq += (*axis)[p] * (*axis)[p];
            }

            let rnorm = rsqrt(norm_sq);
            for (var p = 0u; p < channels; p++) {
                a_vec[p] *= rnorm;
            }
        }
    }

    for (var p = 0u; p < channels; p++) {
        (*axis)[p] = a_vec[p];
    }
}

fn compute_stats_masked(stats: ptr<function, array<f32, 15>>, block: ptr<function, array<f32, 64>>, mask: u32, channels: u32) {
    var mask_shifted = mask << 1u;
    for (var k = 0u; k < 16u; k++) {
        mask_shifted = mask_shifted >> 1u;
        let flag = f32(mask_shifted & 1u);

        var rgba: vec4<f32>;
        for (var p = 0u; p < channels; p++) {
            rgba[p] = (*block)[k + p * 16u] * flag;
        }
        (*stats)[14] += flag;

        (*stats)[10] += rgba[0];
        (*stats)[11] += rgba[1];
        (*stats)[12] += rgba[2];

        (*stats)[0] += rgba[0] * rgba[0];
        (*stats)[1] += rgba[0] * rgba[1];
        (*stats)[2] += rgba[0] * rgba[2];

        (*stats)[4] += rgba[1] * rgba[1];
        (*stats)[5] += rgba[1] * rgba[2];

        (*stats)[7] += rgba[2] * rgba[2];

        if (channels == 4u) {
            (*stats)[13] += rgba[3];
            (*stats)[3] += rgba[0] * rgba[3];
            (*stats)[6] += rgba[1] * rgba[3];
            (*stats)[8] += rgba[2] * rgba[3];
            (*stats)[9] += rgba[3] * rgba[3];
        }
    }
}

fn covar_from_stats(covar: ptr<function, array<f32, 10>>, stats: array<f32, 15>, channels: u32) {
    (*covar)[0] = stats[0] - stats[10] * stats[10] / stats[14];
    (*covar)[1] = stats[1] - stats[10] * stats[11] / stats[14];
    (*covar)[2] = stats[2] - stats[10] * stats[12] / stats[14];

    (*covar)[4] = stats[4] - stats[11] * stats[11] / stats[14];
    (*covar)[5] = stats[5] - stats[11] * stats[12] / stats[14];

    (*covar)[7] = stats[7] - stats[12] * stats[12] / stats[14];

    if (channels == 4u) {
        (*covar)[3] = stats[3] - stats[10] * stats[13] / stats[14];
        (*covar)[6] = stats[6] - stats[11] * stats[13] / stats[14];
        (*covar)[8] = stats[8] - stats[12] * stats[13] / stats[14];
        (*covar)[9] = stats[9] - stats[13] * stats[13] / stats[14];
    }
}

fn compute_covar_dc_masked(covar: ptr<function, array<f32, 10>>, dc: ptr<function, vec4<f32>>, block: ptr<function, array<f32, 64>>, mask: u32, channels: u32) {
    var stats: array<f32, 15>;
    compute_stats_masked(&stats, block, mask, channels);

    // Calculate dc values from stats
    for (var p = 0u; p < channels; p++) {
        (*dc)[p] = stats[10u + p] / stats[14];
    }

    covar_from_stats(covar, stats, channels);
}

fn block_pca_axis(axis: ptr<function, vec4<f32>>, dc: ptr<function, vec4<f32>>, block: ptr<function, array<f32, 64>>, mask: u32, channels: u32) {
    const power_iterations = 8u; // 4 not enough for HQ

    var covar: array<f32, 10>;
    compute_covar_dc_masked(&covar, dc, block, mask, channels);

    const inv_var = 1.0 / (256.0 * 256.0);
    for (var k = 0u; k < 10u; k++) {
        covar[k] *= inv_var;
    }

    let eps = sq(0.001);
    covar[0] += eps;
    covar[4] += eps;
    covar[7] += eps;
    covar[9] += eps;

    compute_axis(axis, &covar, power_iterations, channels);
}

fn block_pca_bound_split(block: ptr<function, array<f32, 64>>, mask: u32, full_stats: array<f32, 15>, channels: u32) -> f32 {
    var stats: array<f32, 15>;
    compute_stats_masked(&stats, block, mask, channels);

    var covar1: array<f32, 10>;
    covar_from_stats(&covar1, stats, channels);

    for (var i = 0u; i < 15u; i++) {
        stats[i] = full_stats[i] - stats[i];
    }

    var covar2: array<f32, 10>;
    covar_from_stats(&covar2, stats, channels);

    var bound = 0.0;
    bound += get_pca_bound(covar1, channels);
    bound += get_pca_bound(covar2, channels);

    return sqrt(bound) * 256.0;
}

fn unpack_to_byte(v: i32, bits: u32) -> i32 {
    let vv = v << (8u - bits);
    return vv + (vv >> bits);
}

fn ep_quant0367(qep:u32, ep:u32, offset: u32, mode: u32, channels: u32) {
    let bits = select(select(7u, 5u, mode == 7u), 4u, mode == 0u);
    let levels = i32(1u << bits);
    let levels2 = levels * 2 - 1;

    for (var i = 0u; i < 2u; i++) {
        var qep_b: array<i32, 8>;

        for (var b = 0u; b < 2u; b++) {
            for (var p = 0u; p < 4u; p++) {
                let v = i32((encoder_f32_24[ep*24u+u32(offset + i * 4 + p)][encoder_lane] / 255.0 * f32(levels2) - f32(b)) / 2.0 + 0.5) * 2 + i32(b);
                qep_b[b * 4 + p] = clamp(v, i32(b), levels2 - 1 + i32(b));
            }
        }

        var ep_b: array<f32, 8>;
        for (var j = 0u; j < 8u; j++) {
            ep_b[j] = f32(qep_b[j]);
        }

        if (mode == 0u) {
            for (var j = 0u; j < 8u; j++) {
                ep_b[j] = f32(unpack_to_byte(qep_b[j], 5u));
            }
        }

        var err0 = 0.0;
        var err1 = 0.0;
        for (var p = 0u; p < channels; p++) {
            err0 += sq(encoder_f32_24[ep*24u+u32(offset + i * 4u + p)][encoder_lane] - ep_b[0u + p]);
            err1 += sq(encoder_f32_24[ep*24u+u32(offset + i * 4u + p)][encoder_lane] - ep_b[4u + p]);
        }

        for (var p = 0u; p < 4u; p++) {
            encoder_i32_24[qep*24u+u32(offset + i * 4u + p)][encoder_lane] = select(qep_b[4u + p], qep_b[0u + p], err0 < err1);
        }
    }
}

fn ep_quant1(qep:u32, ep:u32, offset: u32) {
    var qep_b: array<i32, 16>;

    for (var b = 0u; b < 2u; b++) {
        for (var i = 0u; i < 8u; i++) {
            let v = i32((encoder_f32_24[ep*24u+u32(offset + i)][encoder_lane] / 255.0 * 127.0 - f32(b)) / 2.0 + 0.5) * 2 + i32(b);
            qep_b[b * 8u + i] = clamp(v, i32(b), 126 + i32(b));
        }
    }

    // dequant
    var ep_b: array<f32, 16>;
    for (var k = 0u; k < 16u; k++) {
        ep_b[k] = f32(unpack_to_byte(qep_b[k], 7u));
    }

    var err0 = 0.0;
    var err1 = 0.0;
    for (var j = 0u; j < 2u; j++) {
        for (var p = 0u; p < 3u; p++) {
            err0 += sq(encoder_f32_24[ep*24u+u32(offset + j * 4u + p)][encoder_lane] - ep_b[0u + j * 4u + p]);
            err1 += sq(encoder_f32_24[ep*24u+u32(offset + j * 4u + p)][encoder_lane] - ep_b[8u + j * 4u + p]);
        }
    }

    for (var i = 0u; i < 8u; i++) {
        encoder_i32_24[qep*24u+u32(offset + i)][encoder_lane] = select(qep_b[8u + i], qep_b[0u + i], err0 < err1);
    }
}

fn ep_quant245(qep:u32, ep:u32, offset: u32, mode: u32) {
    let bits = select(5u, 7u, mode == 5u);
    let levels = i32(1u << bits);

    for (var i = 0u; i < 8u; i++) {
        let v = i32(encoder_f32_24[ep*24u+u32(offset + i)][encoder_lane] / 255.0 * f32(levels - 1) + 0.5);
        encoder_i32_24[qep*24u+u32(offset + i)][encoder_lane] = clamp(v, 0, levels - 1);
    }
}

fn ep_quant(qep:u32, ep:u32, mode: u32, channels: u32) {
    const pairs_table = array<u32, 8>(3u, 2u, 3u, 2u, 1u, 1u, 1u, 2u);
    let pairs = pairs_table[mode];

    if (mode == 0u || mode == 3u || mode == 6u || mode == 7u) {
        for (var i = 0u; i < pairs; i++) {
            ep_quant0367(qep, ep, i * 8u, mode, channels);
        }
    } else if (mode == 1u) {
        for (var i = 0u; i < pairs; i++) {
            ep_quant1(qep, ep, i * 8u);
        }
    } else if (mode == 2u || mode == 4u || mode == 5u) {
        for (var i = 0u; i < pairs; i++) {
            ep_quant245(qep, ep, i * 8u, mode);
        }
    }
}

fn ep_dequant(ep:u32, qep:u32, mode: u32) {
    const pairs_table = array<u32, 8>(3u, 2u, 3u, 2u, 1u, 1u, 1u, 2u);
    let pairs = pairs_table[mode];

    // mode 3, 6 are 8-bit
    if (mode == 3u || mode == 6u) {
        for (var i = 0u; i < 8u * pairs; i++) {
            encoder_f32_24[ep*24u+u32(i)][encoder_lane] = f32(encoder_i32_24[qep*24u+u32(i)][encoder_lane]);
        }
    } else if (mode == 1u || mode == 5u) {
        for (var i = 0u; i < 8u * pairs; i++) {
            encoder_f32_24[ep*24u+u32(i)][encoder_lane] = f32(unpack_to_byte(encoder_i32_24[qep*24u+u32(i)][encoder_lane], 7u));
        }
    } else if (mode == 0u || mode == 2u || mode == 4u) {
        for (var i = 0u; i < 8u * pairs; i++) {
            encoder_f32_24[ep*24u+u32(i)][encoder_lane] = f32(unpack_to_byte(encoder_i32_24[qep*24u+u32(i)][encoder_lane], 5u));
        }
    } else if (mode == 7u) {
        for (var i = 0u; i < 8u * pairs; i++) {
            encoder_f32_24[ep*24u+u32(i)][encoder_lane] = f32(unpack_to_byte(encoder_i32_24[qep*24u+u32(i)][encoder_lane], 6u));
        }
    }
}

fn ep_quant_dequant(qep:u32, ep:u32, mode: u32, channels: u32) {
    ep_quant(qep, ep, mode, channels);
    ep_dequant(ep, qep, mode);
}

fn block_quant(qblock: ptr<function, vec2<u32>>, block: ptr<function, array<f32, 64>>, bits: u32, ep:u32, pattern: u32, channels: u32) -> f32 {
    var total_err = 0.0;
    let levels = 1u << bits;

    (*qblock)[0] = 0u;
    (*qblock)[1] = 0u;

    var pattern_shifted = pattern;
    for (var k = 0u; k < 16u; k++) {
        let j = pattern_shifted & 3u;
        pattern_shifted = pattern_shifted >> 2u;

        var proj = 0.0;
        var div = 0.0;
        for (var p = 0u; p < channels; p++) {
            let ep_a = encoder_f32_24[ep*24u+u32(8u * j + 0u + p)][encoder_lane];
            let ep_b = encoder_f32_24[ep*24u+u32(8u * j + 4u + p)][encoder_lane];
            proj += ((*block)[k + p * 16u] - ep_a) * (ep_b - ep_a);
            div += sq(ep_b - ep_a);
        }

        proj = proj / div;

        let q1 = i32(proj * f32(levels) + 0.5);
        let q1_clamped = clamp(q1, 1, i32(levels) - 1);

        var err0 = 0.0;
        var err1 = 0.0;
        let w0 = get_unquant_value(bits, q1_clamped - 1);
        let w1 = get_unquant_value(bits, q1_clamped);

        for (var p = 0u; p < channels; p++) {
            let ep_a = encoder_f32_24[ep*24u+u32(8u * j + 0u + p)][encoder_lane];
            let ep_b = encoder_f32_24[ep*24u+u32(8u * j + 4u + p)][encoder_lane];
            let dec_v0 = f32(((64 - w0) * i32(ep_a) + w0 * i32(ep_b) + 32) / 64);
            let dec_v1 = f32(((64 - w1) * i32(ep_a) + w1 * i32(ep_b) + 32) / 64);
            err0 += sq(dec_v0 - (*block)[k + p * 16u]);
            err1 += sq(dec_v1 - (*block)[k + p * 16u]);
        }

        var best_err = err1;
        var best_q = q1_clamped;
        if (err0 < err1) {
            best_err = err0;
            best_q = q1_clamped - 1;
        }

        (*qblock)[k / 8u] |= u32(best_q) << (4u * (k % 8u));
        total_err += best_err;
    }

    return total_err;
}

fn block_segment_core(ep:u32, offset: u32, block: ptr<function, array<f32, 64>>, mask: u32, channels: u32) {
    var axis: vec4<f32>;
    var dc: vec4<f32>;
    block_pca_axis(&axis, &dc, block, mask, channels);

    var ext = vec2<f32>(3.402823e38, -3.402823e38);

    // Find min/max
    var mask_shifted = mask << 1u;
    for (var k = 0u; k < 16u; k++) {
        mask_shifted = mask_shifted >> 1u;
        if ((mask_shifted & 1u) == 0u) {
            continue;
        }

        var dot = 0.0;
        for (var p = 0u; p < channels; p++) {
            dot += axis[p] * ((*block)[16u * p + k] - dc[p]);
        }

        ext[0] = min(ext[0], dot);
        ext[1] = max(ext[1], dot);
    }

    // Create some distance if the endpoints collapse
    if (ext[1] - ext[0] < 1.0) {
        ext[0] -= 0.5;
        ext[1] += 0.5;
    }

    for (var i = 0u; i < 2u; i++) {
        for (var p = 0u; p < channels; p++) {
            encoder_f32_24[ep*24u+u32(offset + 4u * i + p)][encoder_lane] = ext[i] * axis[p] + dc[p];
        }
    }
}

fn block_segment(ep:u32, offset: u32, block: ptr<function, array<f32, 64>>, mask: u32, channels: u32) {
    block_segment_core(ep, offset, block, mask, channels);

    for (var i = 0u; i < 2u; i++) {
        for (var p = 0u; p < channels; p++) {
            encoder_f32_24[ep*24u+u32(offset + 4u * i + p)][encoder_lane] = clamp(encoder_f32_24[ep*24u+u32(offset + 4u * i + p)][encoder_lane], 0.0, 255.0);
        }
    }
}

fn opt_channel(qblock: ptr<function, vec2<u32>>, qep: ptr<function, vec2<i32>>, channel_block: ptr<function, array<f32, 16>>, bits: u32, epbits: u32) -> f32 {
    var ep: vec2<f32> = vec2<f32>(255.0, 0.0);

    for (var k = 0u; k < 16u; k++) {
        ep[0] = min(ep[0], (*channel_block)[k]);
        ep[1] = max(ep[1], (*channel_block)[k]);
    }

    channel_quant_dequant(qep, &ep, epbits);
    var err = channel_opt_quant(qblock, channel_block, bits, &ep);

    // Refine
    let refine_iterations = settings.refine_iterations_channel;
    for (var i = 0u; i < refine_iterations; i++) {
        channel_opt_endpoints(&ep, channel_block, bits, *qblock);
        channel_quant_dequant(qep, &ep, epbits);
        err = channel_opt_quant(qblock, channel_block, bits, &ep);
    }

    return err;
}

fn channel_quant_dequant(qep: ptr<function, vec2<i32>>, ep: ptr<function, vec2<f32>>, epbits: u32) {
    let elevels = i32(1u << epbits);

    for (var i = 0u; i < 2u; i++) {
        let v = i32((*ep)[i] / 255.0 * f32(elevels - 1) + 0.5);
        (*qep)[i] = clamp(v, 0, elevels - 1);
        (*ep)[i] = f32(unpack_to_byte((*qep)[i], epbits));
    }
}

fn channel_opt_quant(qblock: ptr<function, vec2<u32>>, channel_block: ptr<function, array<f32, 16>>, bits: u32, ep: ptr<function, vec2<f32>>) -> f32 {
    let levels = i32(1u << bits);

    (*qblock)[0] = 0u;
    (*qblock)[1] = 0u;

    var total_err = 0.0;

    for (var k = 0u; k < 16u; k++) {
        let proj = ((*channel_block)[k] - (*ep)[0]) / ((*ep)[1] - (*ep)[0] + 0.001);

        let q1 = i32(proj * f32(levels) + 0.5);
        let q1_clamped = clamp(q1, 1, levels - 1);

        var err0 = 0.0;
        var err1 = 0.0;
        let w0 = get_unquant_value(bits, q1_clamped - 1);
        let w1 = get_unquant_value(bits, q1_clamped);

        let dec_v0 = f32(((64 - w0) * i32((*ep)[0]) + w0 * i32((*ep)[1]) + 32) / 64);
        let dec_v1 = f32(((64 - w1) * i32((*ep)[0]) + w1 * i32((*ep)[1]) + 32) / 64);
        err0 += sq(dec_v0 - (*channel_block)[k]);
        err1 += sq(dec_v1 - (*channel_block)[k]);

        let best_err = select(err1, err0, err0 < err1);
        let best_q = select(q1_clamped, q1_clamped - 1, err0 < err1);

        (*qblock)[k / 8u] |= u32(best_q) << (4u * (k % 8u));
        total_err += best_err;
    }

    return total_err;
}

fn channel_opt_endpoints(ep: ptr<function, vec2<f32>>, channel_block: ptr<function, array<f32, 16>>, bits: u32, qblock: vec2<u32>) {
    let levels = i32(1u << bits);

    var Atb1 = 0.0;
    var sum_q = 0.0;
    var sum_qq = 0.0;
    var sum = 0.0;

    for (var k1 = 0u; k1 < 2u; k1++) {
        var qbits_shifted = qblock[k1];
        for (var k2 = 0u; k2 < 8u; k2++) {
            let k = k1 * 8u + k2;
            let q = f32(qbits_shifted & 15u);
            qbits_shifted >>= 4u;

            let x = f32(levels - 1) - q;

            sum_q += q;
            sum_qq += q * q;

            sum += (*channel_block)[k];
            Atb1 += x * (*channel_block)[k];
        }
    }

    let Atb2 = f32(levels - 1) * sum - Atb1;

    let Cxx = 16.0 * sq(f32(levels - 1)) - 2.0 * f32(levels - 1) * sum_q + sum_qq;
    let Cyy = sum_qq;
    let Cxy = f32(levels - 1) * sum_q - sum_qq;
    let scale = f32(levels - 1) / (Cxx * Cyy - Cxy * Cxy);

    (*ep)[0] = (Atb1 * Cyy - Atb2 * Cxy) * scale;
    (*ep)[1] = (Atb2 * Cxx - Atb1 * Cxy) * scale;

    (*ep)[0] = clamp((*ep)[0], 0.0, 255.0);
    (*ep)[1] = clamp((*ep)[1], 0.0, 255.0);

    if (abs(Cxx * Cyy - Cxy * Cxy) < 0.001) {
        (*ep)[0] = sum / 16.0;
        (*ep)[1] = (*ep)[0];
    }
}

fn partial_sort_list(list:u32, length: i32, partial_count: i32) {
    for (var k = 0; k < partial_count; k++) {
        var best_idx = i32(k);
        var best_value = encoder_i32_64[list*64u+u32(k)][encoder_lane];

        for (var i = k + 1; i < length; i++) {
            if (best_value > encoder_i32_64[list*64u+u32(i)][encoder_lane]) {
                best_value = encoder_i32_64[list*64u+u32(i)][encoder_lane];
                best_idx = i;
            }
        }

        let temp = encoder_i32_64[list*64u+u32(k)][encoder_lane];
        encoder_i32_64[list*64u+u32(k)][encoder_lane] = best_value;
        encoder_i32_64[list*64u+u32(best_idx)][encoder_lane] = temp;
    }
}

fn put_bits(state: ptr<function, State>, pos: ptr<function, u32>, bits: u32, v: u32) {
    (*state).data[(*pos) / 32u] |= v << ((*pos) % 32u);
    if ((*pos) % 32u + bits > 32u) {
        (*state).data[(*pos) / 32u + 1u] |= v >> (32u - (*pos) % 32u);
    }
    *pos += bits;
}

fn data_shl_1bit_from(state: ptr<function, State>, from_bits: u32) {
    if (from_bits < 96u) {
        let shifted = ((*state).data[2] >> 1u) | ((*state).data[3] << 31u);
        let mask = ((1u << (from_bits - 64u)) - 1u) >> 1u;
        (*state).data[2] = (mask & (*state).data[2]) | (~mask & shifted);
        (*state).data[3] = ((*state).data[3] >> 1u) | ((*state).data[4] << 31u);
        (*state).data[4] = (*state).data[4] >> 1u;
    } else if (from_bits < 128u) {
        let shifted = ((*state).data[3] >> 1u) | ((*state).data[4] << 31u);
        let mask = ((1u << (from_bits - 96u)) - 1u) >> 1u;
        (*state).data[3] = (mask & (*state).data[3]) | (~mask & shifted);
        (*state).data[4] = (*state).data[4] >> 1u;
    }
}

fn opt_endpoints(ep:u32, offset: u32, block: ptr<function, array<f32, 64>>, bits: u32, qblock: vec2<u32>, mask: u32, channels: u32) {
    let levels = i32(1u << bits);

    var Atb1: vec4<f32>;
    var sum_q = 0.0;
    var sum_qq = 0.0;
    var sum: array<f32, 5>;

    var mask_shifted = mask << 1u;
    for (var k1 = 0u; k1 < 2u; k1++) {
        var qbits_shifted = qblock[k1];
        for (var k2 = 0u; k2 < 8u; k2++) {
            let k = k1 * 8u + k2;
            let q = f32(qbits_shifted & 15u);
            qbits_shifted >>= 4u;

            mask_shifted >>= 1u;
            if ((mask_shifted & 1u) == 0u) {
                continue;
            }

            let x = f32(levels - 1) - q;

            sum_q += q;
            sum_qq += q * q;

            sum[4] += 1.0;
            for (var p = 0u; p < channels; p++) {
                sum[p] += (*block)[k + p * 16u];
                Atb1[p] += x * (*block)[k + p * 16u];
            }
        }
    }

    var Atb2: vec4<f32>;
    for (var p = 0u; p < channels; p++) {
        Atb2[p] = f32(levels - 1) * sum[p] - Atb1[p];
    }

    let Cxx = sum[4] * sq(f32(levels - 1)) - 2.0 * f32(levels - 1) * sum_q + sum_qq;
    let Cyy = sum_qq;
    let Cxy = f32(levels - 1) * sum_q - sum_qq;
    let scale = f32(levels - 1) / (Cxx * Cyy - Cxy * Cxy);

    for (var p = 0u; p < channels; p++) {
        encoder_f32_24[ep*24u+u32(offset + 0u + p)][encoder_lane] = (Atb1[p] * Cyy - Atb2[p] * Cxy) * scale;
        encoder_f32_24[ep*24u+u32(offset + 4u + p)][encoder_lane] = (Atb2[p] * Cxx - Atb1[p] * Cxy) * scale;
    }

    if (abs(Cxx * Cyy - Cxy * Cxy) < 0.001) {
        // flatten
        for (var p = 0u; p < channels; p++) {
            encoder_f32_24[ep*24u+u32(offset + 0u + p)][encoder_lane] = sum[p] / sum[4];
            encoder_f32_24[ep*24u+u32(offset + 4u + p)][encoder_lane] = encoder_f32_24[ep*24u+u32(offset + 0u + p)][encoder_lane];
        }
    }
}

fn bc7_code_qblock(state: ptr<function, State>, qpos: ptr<function, u32>, qblock: vec2<u32>, bits: u32, flips: u32) {
    let levels = 1u << bits;
    var flips_shifted = flips;

    for (var k1 = 0u; k1 < 2u; k1++) {
        var qbits_shifted = qblock[k1];
        for (var k2 = 0u; k2 < 8u; k2++) {
            var q = qbits_shifted & 15u;
            if ((flips_shifted & 1u) > 0u) {
                q = (levels - 1u) - q;
            }

            if (k1 == 0u && k2 == 0u) {
                put_bits(state, qpos, bits - 1u, q);
            } else {
                put_bits(state, qpos, bits, q);
            }
            qbits_shifted >>= 4u;
            flips_shifted >>= 1u;
        }
    }
}

fn bc7_code_adjust_skip_mode01237(state: ptr<function, State>, mode: u32, part_id: i32) {
    let pairs = select(2u, 3u, mode == 0u || mode == 2u);
    let bits = select(2u, 3u, mode == 0u || mode == 1u);

    var skips = get_skips(part_id);

    if (pairs > 2u && skips[1] < skips[2]) {
        let t = skips[1];
        skips[1] = skips[2];
        skips[2] = t;
    }

    for (var j = 1u; j < pairs; j++) {
        let k = skips[j];
        data_shl_1bit_from(state, 128u + (pairs - 1u) - (15u - k) * bits);
    }
}

fn bc7_code_apply_swap_mode456(qep:u32, channels: u32, qblock: ptr<function, vec2<u32>>, bits: u32) {
    let levels = 1u << bits;

    if (((*qblock)[0] & 15u) >= levels / 2u) {
        for (var p = 0u; p < channels; p++) {
            let temp = encoder_i32_24[qep*24u+u32(p)][encoder_lane];
            encoder_i32_24[qep*24u+u32(p)][encoder_lane] = encoder_i32_24[qep*24u+u32(channels + p)][encoder_lane];
            encoder_i32_24[qep*24u+u32(channels + p)][encoder_lane] = temp;
        }

        for (var k = 0u; k < 2u; k++) {
            (*qblock)[k] = (0x11111111u * (levels - 1u)) - (*qblock)[k];
        }
    }
}

fn bc7_code_apply_swap_mode01237(qep:u32, qblock: vec2<u32>, mode: u32, part_id: i32) -> u32 {
    let bits = select(2u, 3u, mode == 0u || mode == 1u);
    let pairs = select(2u, 3u, mode == 0u || mode == 2u);

    var flips = 0u;
    let levels = 1u << bits;

    let skips = get_skips(part_id);

    for (var j = 0u; j < pairs; j++) {
        let k0 = skips[j];
        // Extract 4 bits from qblock at position k0
        let q = (qblock[k0 >> 3u] << (28u - (k0 & 7u) * 4u)) >> 28u;

        if (q >= levels / 2u) {
            for (var p = 0u; p < 4u; p++) {
                let temp = encoder_i32_24[qep*24u+u32(8u * j + p)][encoder_lane];
                encoder_i32_24[qep*24u+u32(8u * j + p)][encoder_lane] = encoder_i32_24[qep*24u+u32(8u * j + 4u + p)][encoder_lane];
                encoder_i32_24[qep*24u+u32(8u * j + 4u + p)][encoder_lane] = temp;
            }

            let pmask = get_pattern_mask(part_id, j);
            flips |= u32(pmask);
        }
    }

    return flips;
}

fn bc7_code_mode01237(state: ptr<function, State>, qep:u32, qblock: vec2<u32>, part_id: i32, mode: u32) {
    let bits = select(2u, 3u, mode == 0u || mode == 1u);
    let pairs = select(2u, 3u, mode == 0u || mode == 2u);
    let channels = select(3u, 4u, mode == 7u);

    let flips = bc7_code_apply_swap_mode01237(qep, qblock, mode, part_id);

    for (var k = 0u; k < 5u; k++) {
        (*state).data[k] = 0u;
    }

    var pos = 0u;

    // Mode 0-3, 7
    put_bits(state, &pos, mode + 1u, 1u << mode);

    // Partition
    if (mode == 0u) {
        put_bits(state, &pos, 4u, u32(part_id & 15));
    } else {
        put_bits(state, &pos, 6u, u32(part_id & 63));
    }

    // Endpoints
    for (var p = 0u; p < channels; p++) {
        for (var j = 0u; j < pairs * 2u; j++) {
            if (mode == 0u) {
                put_bits(state, &pos, 4u, u32(encoder_i32_24[qep*24u+u32(j * 4u + p)][encoder_lane]) >> 1u);
            } else if (mode == 1u) {
                put_bits(state, &pos, 6u, u32(encoder_i32_24[qep*24u+u32(j * 4u + p)][encoder_lane]) >> 1u);
            } else if (mode == 2u) {
                put_bits(state, &pos, 5u, u32(encoder_i32_24[qep*24u+u32(j * 4u + p)][encoder_lane]));
            } else if (mode == 3u) {
                put_bits(state, &pos, 7u, u32(encoder_i32_24[qep*24u+u32(j * 4u + p)][encoder_lane]) >> 1u);
            } else if (mode == 7u) {
                put_bits(state, &pos, 5u, u32(encoder_i32_24[qep*24u+u32(j * 4u + p)][encoder_lane]) >> 1u);
            }
        }
    }

    // P bits
    if (mode == 1u) {
        for (var j = 0u; j < 2u; j++) {
            put_bits(state, &pos, 1u, u32(encoder_i32_24[qep*24u+u32(j * 8u)][encoder_lane]) & 1u);
        }
    }

    if (mode == 0u || mode == 3u || mode == 7u) {
        for (var j = 0u; j < pairs * 2u; j++) {
            put_bits(state, &pos, 1u, u32(encoder_i32_24[qep*24u+u32(j * 4u)][encoder_lane]) & 1u);
        }
    }

    // Quantized values
    bc7_code_qblock(state, &pos, qblock, bits, flips);
    bc7_code_adjust_skip_mode01237(state, mode, part_id);
}

fn bc7_code_mode45(state: ptr<function, State>, params: ptr<function, Mode45Parameters>, mode: u32) {
    for(var clear_qep=0u;clear_qep<24u;clear_qep++){encoder_i32_24[0u*24u+u32(clear_qep)][encoder_lane]=0;}
    var qblock: vec2<u32>;
    for(var clear_aqep=0u;clear_aqep<24u;clear_aqep++){encoder_i32_24[1u*24u+u32(clear_aqep)][encoder_lane]=0;}
    var aqblock: vec2<u32>;

    for (var i = 0u; i < 8u; i++) {
        encoder_i32_24[0u*24u+u32(i)][encoder_lane] = (*params).qep[i];
    }
    for (var i = 0u; i < 2u; i++) {
        qblock[i] = (*params).qblock[i];
        encoder_i32_24[1u*24u+u32(i)][encoder_lane] = (*params).aqep[i];
        aqblock[i] = (*params).aqblock[i];
    }
    let rotation = (*params).rotation;
    let swap = (*params).swap;

    let bits = 2u;
    let abits = select(2u, 3u, mode == 4u);
    let epbits = select(7u, 5u, mode == 4u);
    let aepbits = select(8u, 6u, mode == 4u);

    if (swap == 0) {
        bc7_code_apply_swap_mode456(0u, 4u, &qblock, bits);
        bc7_code_apply_swap_mode456(1u, 1u, &aqblock, abits);
    } else {
        // Swap qblock and aqblock
        let temp_block = qblock;
        qblock = aqblock;
        aqblock = temp_block;

        bc7_code_apply_swap_mode456(1u, 1u, &qblock, bits);
        bc7_code_apply_swap_mode456(0u, 4u, &aqblock, abits);
    }

    // Clear state data
    for (var k = 0u; k < 5u; k++) {
        (*state).data[k] = 0u;
    }
    var pos = 0u;

    // Mode 4-5
    put_bits(state, &pos, mode + 1u, 1u << mode);

    // Rotation
    put_bits(state, &pos, 2u, (rotation + 1u) & 3u);

    if (mode == 4u) {
        put_bits(state, &pos, 1u, swap);
    }

    // Endpoints
    for (var p = 0u; p < 3u; p++) {
        put_bits(state, &pos, epbits, u32(encoder_i32_24[0u*24u+u32(0u + p)][encoder_lane]));
        put_bits(state, &pos, epbits, u32(encoder_i32_24[0u*24u+u32(4u + p)][encoder_lane]));
    }

    // Alpha endpoints
    put_bits(state, &pos, aepbits, u32(encoder_i32_24[1u*24u+u32(0)][encoder_lane]));
    put_bits(state, &pos, aepbits, u32(encoder_i32_24[1u*24u+u32(1)][encoder_lane]));

    // Quantized values
    bc7_code_qblock(state, &pos, qblock, bits, 0u);
    bc7_code_qblock(state, &pos, aqblock, abits, 0u);
}

fn bc7_code_mode6(state: ptr<function, State>, qep:u32, qblock: ptr<function, vec2<u32>>) {
    bc7_code_apply_swap_mode456(qep, 4u, qblock, 4u);

    for (var k = 0u; k < 5u; k++) {
        (*state).data[k] = 0u;
    }
    var pos = 0u;

    // Mode 6
    put_bits(state, &pos, 7u, 64u);

    // Endpoints
    for (var p = 0u; p < 4u; p++) {
        put_bits(state, &pos, 7u, u32(encoder_i32_24[qep*24u+u32(0u + p)][encoder_lane]) >> 1u);
        put_bits(state, &pos, 7u, u32(encoder_i32_24[qep*24u+u32(4u + p)][encoder_lane]) >> 1u);
    }

    // P bits
    put_bits(state, &pos, 1u, u32(encoder_i32_24[qep*24u+u32(0)][encoder_lane]) & 1u);
    put_bits(state, &pos, 1u, u32(encoder_i32_24[qep*24u+u32(4)][encoder_lane]) & 1u);

    // Quantized values
    bc7_code_qblock(state, &pos, *qblock, 4u, 0u);
}

fn bc7_enc_mode01237_part_fast(qep:u32, qblock: ptr<function, vec2<u32>>, block: ptr<function, array<f32, 64>>, part_id: i32, mode: u32) -> f32 {
    let pattern = get_pattern(part_id);
    let bits = select(2u, 3u, mode == 0u || mode == 1u);
    let pairs = select(2u, 3u, mode == 0u || mode == 2u);
    let channels = select(3u, 4u, mode == 7u);

    for(var clear_ep=0u;clear_ep<24u;clear_ep++){encoder_f32_24[0u*24u+u32(clear_ep)][encoder_lane]=0.;}
    for (var j = 0u; j < pairs; j++) {
        let mask = get_pattern_mask(part_id, j);
        block_segment(0u, j * 8, block, mask, channels);
    }

    ep_quant_dequant(qep,0u, mode, channels);

    return block_quant(qblock, block, bits,0u, pattern, channels);
}

fn bc7_enc_mode01237(state: ptr<function, State>, block: ptr<function, array<f32, 64>>, mode: u32, part_list:u32, part_count: u32) {
    if (part_count == 0u) {
        return;
    }

    let bits = select(2u, 3u, mode == 0u || mode == 1u);
    let pairs = select(2u, 3u, mode == 0u || mode == 2u);
    let channels = select(3u, 4u, mode == 7u);

    for(var clear_best_qep=0u;clear_best_qep<24u;clear_best_qep++){encoder_i32_24[0u*24u+u32(clear_best_qep)][encoder_lane]=0;}
    var best_qblock: vec2<u32>;
    var best_part_id = -1;
    var best_err = 3.402823e38;

    for (var part = 0u; part < part_count; part++) {
        var part_id = encoder_i32_64[part_list*64u+u32(part)][encoder_lane] & 63;
        part_id = select(part_id, part_id + 64, pairs == 3);

        for(var clear_qep=0u;clear_qep<24u;clear_qep++){encoder_i32_24[1u*24u+u32(clear_qep)][encoder_lane]=0;}
        var qblock: vec2<u32>;
        let err = bc7_enc_mode01237_part_fast(1u, &qblock, block, part_id, mode);

        if (err < best_err) {
            for (var i = 0u; i < 8u * pairs; i++) {
                encoder_i32_24[0u*24u+u32(i)][encoder_lane] = encoder_i32_24[1u*24u+u32(i)][encoder_lane];
            }
            for (var k = 0u; k < 2u; k++) {
                best_qblock[k] = qblock[k];
            }

            best_part_id = part_id;
            best_err = err;
        }
    }

    let refine_iterations = settings.refine_iterations[mode];
    for (var i = 0u; i < refine_iterations; i++) {
        for(var clear_ep=0u;clear_ep<24u;clear_ep++){encoder_f32_24[1u*24u+u32(clear_ep)][encoder_lane]=0.;}
        for (var j = 0u; j < pairs; j++) {
            let mask = get_pattern_mask(best_part_id, j);
            opt_endpoints(1u, j * 8, block, bits, best_qblock, mask, channels);
        }

        for(var clear_qep=0u;clear_qep<24u;clear_qep++){encoder_i32_24[1u*24u+u32(clear_qep)][encoder_lane]=0;}
        var qblock: vec2<u32>;

        ep_quant_dequant(1u,1u, mode, channels);

        let pattern = get_pattern(best_part_id);
        let err = block_quant(&qblock, block, bits,1u, pattern, channels);

        if (err < best_err) {
            for (var i = 0u; i < 8u * pairs; i++) {
                encoder_i32_24[0u*24u+u32(i)][encoder_lane] = encoder_i32_24[1u*24u+u32(i)][encoder_lane];
            }
            for (var k = 0u; k < 2u; k++) {
                best_qblock[k] = qblock[k];
            }

            best_err = err;
        }
    }

    if (mode != 7u) {
        best_err += (*state).opaque_err;
    }

    if (best_err < (*state).best_err) {
        (*state).best_err = best_err;
        bc7_code_mode01237(state,0u, best_qblock, best_part_id, mode);
    }
}

fn bc7_enc_mode02(state: ptr<function, State>, block: ptr<function, array<f32, 64>>) {
    for(var clear_part_list=0u;clear_part_list<64u;clear_part_list++){encoder_i32_64[0u*64u+u32(clear_part_list)][encoder_lane]=0;}
    for (var part = 0; part < 64; part++) {
        encoder_i32_64[0u*64u+u32(part)][encoder_lane] = part;
    }

    bc7_enc_mode01237(state, block, 0u,0u, 16u);

    if (settings.skip_mode2 == 0) {
        bc7_enc_mode01237(state, block, 2u,0u, 64u);
    }
}

fn bc7_enc_mode13(state: ptr<function, State>, block: ptr<function, array<f32, 64>>) {
    if (settings.fast_skip_threshold_mode1 == 0u && settings.fast_skip_threshold_mode3 == 0u) {
        return;
    }

    var full_stats: array<f32, 15>;
    compute_stats_masked(&full_stats, block, 0xFFFFFFFFu, 3u);

    for(var clear_part_list=0u;clear_part_list<64u;clear_part_list++){encoder_i32_64[0u*64u+u32(clear_part_list)][encoder_lane]=0;}
    for (var part = 0; part < 64; part++) {
        let mask = get_pattern_mask(part, 0u);
        let bound12 = block_pca_bound_split(block, mask, full_stats, 3u);
        let bound = i32(bound12);
        encoder_i32_64[0u*64u+u32(part)][encoder_lane] = part + bound * 64;
    }

    let partial_count = max(settings.fast_skip_threshold_mode1, settings.fast_skip_threshold_mode3);
    partial_sort_list(0u, 64, i32(partial_count));
    bc7_enc_mode01237(state, block, 1u,0u, settings.fast_skip_threshold_mode1);
    bc7_enc_mode01237(state, block, 3u,0u, settings.fast_skip_threshold_mode3);
}

fn bc7_enc_mode45_candidate(best_candidate: ptr<function, Mode45Parameters>, best_err: ptr<function, f32>, block: ptr<function, array<f32, 64>>, mode: u32, rotation: u32, swap: u32) {
    var bits = 2u;
    var abits = 2u;
    var aepbits = 8u;

    if (mode == 4u) {
        abits = 3u;
        aepbits = 6u;
    }

    // (mode 4)
    if (swap == 1u) {
        bits = 3u;
        abits = 2u;
    }

    var candidate_block: array<f32, 64>;

    for (var k = 0u; k < 16u; k++) {
        for (var p = 0u; p < 3u; p++) {
            candidate_block[k + p * 16u] = (*block)[k + p * 16u];
        }

        if (rotation < 3u) {
            // Apply channel rotation
            if (settings.channels == 4u) {
                candidate_block[k + rotation * 16u] = (*block)[k + 3u * 16u];
            }
            if (settings.channels == 3u) {
                candidate_block[k + rotation * 16u] = 255.0;
            }
        }
    }

    for(var clear_ep=0u;clear_ep<24u;clear_ep++){encoder_f32_24[0u*24u+u32(clear_ep)][encoder_lane]=0.;}
    block_segment(0u, 0u, &candidate_block, 0xFFFFFFFFu, 3u);

    for(var clear_qep=0u;clear_qep<24u;clear_qep++){encoder_i32_24[0u*24u+u32(clear_qep)][encoder_lane]=0;}
    ep_quant_dequant(0u,0u, mode, 3u);

    var qblock: vec2<u32>;
    var err = block_quant(&qblock, &candidate_block, bits,0u, 0u, 3u);

    // Refine
    let refine_iterations = settings.refine_iterations[mode];
    for (var i = 0u; i < refine_iterations; i++) {
        opt_endpoints(0u, 0u, &candidate_block, bits, qblock, 0xFFFFFFFFu, 3u);
        ep_quant_dequant(0u,0u, mode, 3u);
        err = block_quant(&qblock, &candidate_block, bits,0u, 0u, 3u);
    }

    var channel_data: array<f32, 16>;
    for (var k = 0u; k < 16u; k++) {
        channel_data[k] = (*block)[k + rotation * 16u];
    }

    // Encoding selected channel
    var aqep: vec2<i32>;
    var aqblock: vec2<u32>;

    err += opt_channel(&aqblock, &aqep, &channel_data, abits, aepbits);

    if (err < *best_err) {
        for (var i = 0u; i < 8u; i++) {
            (*best_candidate).qep[i] = encoder_i32_24[0u*24u+u32(i)][encoder_lane];
        }
        for (var i = 0u; i < 2u; i++) {
            (*best_candidate).qblock[i] = qblock[i];
            (*best_candidate).aqblock[i] = aqblock[i];
            (*best_candidate).aqep[i] = aqep[i];
        }
        (*best_candidate).rotation = rotation;
        (*best_candidate).swap = swap;
        *best_err = err;
    }
}

fn bc7_enc_mode45(state: ptr<function, State>, block: ptr<function, array<f32, 64>>) {
    var best_candidate: Mode45Parameters;
    var best_err = (*state).best_err;

    let channel0 = settings.mode45_channel0;
    for (var p = channel0; p < settings.channels; p++) {
        bc7_enc_mode45_candidate(&best_candidate, &best_err, block, 4u, p, 0u);
        bc7_enc_mode45_candidate(&best_candidate, &best_err, block, 4u, p, 1u);
    }

    // Mode 4
    if (best_err < (*state).best_err) {
        (*state).best_err = best_err;
        bc7_code_mode45(state, &best_candidate, 4u);
    }

    for (var p = channel0; p < settings.channels; p++) {
        bc7_enc_mode45_candidate(&best_candidate, &best_err, block, 5u, p, 0u);
    }

    // Mode 5
    if (best_err < (*state).best_err) {
        (*state).best_err = best_err;
        bc7_code_mode45(state, &best_candidate, 5u);
    }
}

fn bc7_enc_mode6(state: ptr<function, State>, block: ptr<function, array<f32, 64>>) {
    const mode = 6u;
    const bits = 4u;

    for(var clear_ep=0u;clear_ep<24u;clear_ep++){encoder_f32_24[0u*24u+u32(clear_ep)][encoder_lane]=0.;}
    block_segment(0u, 0u, block, 0xFFFFFFFFu, settings.channels);

    if (settings.channels == 3u) {
        encoder_f32_24[0u*24u+u32(3)][encoder_lane] = 255.0;
        encoder_f32_24[0u*24u+u32(7)][encoder_lane] = 255.0;
    }

    for(var clear_qep=0u;clear_qep<24u;clear_qep++){encoder_i32_24[0u*24u+u32(clear_qep)][encoder_lane]=0;}
    ep_quant_dequant(0u,0u, mode, settings.channels);

    var qblock: vec2<u32>;
    var err = block_quant(&qblock, block, bits,0u, 0u, settings.channels);

    let refine_iterations = settings.refine_iterations[mode];
    for (var i = 0u; i < refine_iterations; i++) {
        opt_endpoints(0u, 0u, block, bits, qblock, 0xFFFFFFFFu, settings.channels);
        ep_quant_dequant(0u,0u, mode, settings.channels);
        err = block_quant(&qblock, block, bits,0u, 0u, settings.channels);
    }

    if (err < (*state).best_err) {
        (*state).best_err = err;
        bc7_code_mode6(state,0u, &qblock);
    }
}

fn bc7_enc_mode7(state: ptr<function, State>, block: ptr<function, array<f32, 64>>) {
    if (settings.fast_skip_threshold_mode7 == 0u) {
        return;
    }

    var full_stats: array<f32, 15>;
    compute_stats_masked(&full_stats, block, 0xFFFFFFFFu, settings.channels);

    for(var clear_part_list=0u;clear_part_list<64u;clear_part_list++){encoder_i32_64[0u*64u+u32(clear_part_list)][encoder_lane]=0;}
    for (var part = 0; part < 64; part++) {
        let mask = get_pattern_mask(part, 0u);
        let bound12 = block_pca_bound_split(block, mask, full_stats, settings.channels);
        let bound = i32(bound12);
        encoder_i32_64[0u*64u+u32(part)][encoder_lane] = part + bound * 64;
    }

    partial_sort_list(0u, 64, i32(settings.fast_skip_threshold_mode7));
    bc7_enc_mode01237(state, block, 7u,0u, settings.fast_skip_threshold_mode7);
}

fn compress_block_bc7_core(state: ptr<function, State>, block: ptr<function, array<f32, 64>>) {
    if (settings.mode_selection[0] != 0u) {
        bc7_enc_mode02(state, block);
    }
    if (settings.mode_selection[1] != 0u) {
        bc7_enc_mode13(state,block);
        bc7_enc_mode7(state, block);
    }
    if (settings.mode_selection[2] != 0u) {
        bc7_enc_mode45(state, block);
    }
    if (settings.mode_selection[3] != 0u) {
        bc7_enc_mode6(state, block);
    }
}

fn compute_opaque_err(block: ptr<function, array<f32, 64>>) -> f32 {
    if (settings.channels == 3u) {
        return 0.0;
    } else {
        var err = 0.0;
        for (var k = 0u; k < 16u; k++) {
            err += sq((*block)[48u + k] - 255.0);
        }
        return err;
    }
}

@compute
@workgroup_size(32, 1)
fn compress_bc7(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_index) lane:u32) {
encoder_lane=lane;

    let xx = global_id.x;
    let yy = global_id.y;

    let block_width = (uniforms.width + 3u) / 4u;
    let block_height = (uniforms.height + 3u) / 4u;

    if (xx >= block_width || yy >= block_height) {
        return;
    }

    var block: array<f32, 64>;

    load_block_interleaved_rgba(&block, xx, yy);

    var state: State;
    state.best_err = 3.402823e38;
    state.opaque_err = compute_opaque_err(&block);

    compress_block_bc7_core(&state, &block);

    store_data(&state, block_width, xx, yy);
}

var<private> encoder_lane:u32;
var<workgroup> encoder_i32_24:array<array<i32,32>,48>;
var<workgroup> encoder_f32_24:array<array<f32,32>,48>;
var<workgroup> encoder_i32_64:array<array<i32,32>,64>;
