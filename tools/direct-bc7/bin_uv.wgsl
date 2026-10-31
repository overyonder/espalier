@group(0) @binding(0) var<storage,read> uv_data:array<f32>;
@group(0) @binding(1) var<storage,read> index_data:array<u32>;
@group(0) @binding(2) var<storage,read> offsets:array<u32>;
@group(0) @binding(3) var<storage,read_write> triangles:array<u32>;
@group(0) @binding(4) var<storage,read_write> counts:array<atomic<u32>>;
@group(0) @binding(5) var<uniform> parameters:vec4<u32>;
fn vertex_index(i:u32)->u32 {
 if(parameters.z==2u){return (index_data[i/2u]>>((i%2u)*16u))&65535u;}return index_data[i];
}
fn vertex_uv(i:u32)->vec2<f32>{return vec2(uv_data[i*2u],uv_data[i*2u+1u])*4096.;}
@compute @workgroup_size(128)
fn bin_uv(@builtin(global_invocation_id) invocation:vec3<u32>){
 let face=invocation.x;if(face>=parameters.x){return;}
 let a=vertex_uv(vertex_index(face*3u));let b=vertex_uv(vertex_index(face*3u+1u));let c=vertex_uv(vertex_index(face*3u+2u));
 let lo=vec2<u32>(clamp(floor((min(a,min(b,c))-4.)/32.),vec2(0.),vec2(127.)));
 let hi=vec2<u32>(clamp(floor((max(a,max(b,c))+4.)/32.),vec2(0.),vec2(127.)));
 for(var y=lo.y;y<=hi.y;y++){for(var x=lo.x;x<=hi.x;x++){
  let bin=y*128u+x;let slot=atomicAdd(&counts[bin],1u);
  if(parameters.y!=0u){triangles[offsets[bin]+slot]=face;}
 }}
}
