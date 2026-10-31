@group(0) @binding(0) var<storage,read> surface:array<vec4<f32>>;
@group(0) @binding(1) var<storage,read_write> output:array<u32>;
@group(0) @binding(2) var atlas:texture_2d<f32>;
@group(0) @binding(3) var linear_sampler:sampler;
@compute @workgroup_size(8,8)
fn preview(@builtin(global_invocation_id) id:vec3<u32>){
 if(id.x>=1080u||id.y>=1080u){return;}
 let i=id.y*1080u+id.x;let s=surface[i];var colour=vec4(0.,0.,0.,1.);
 if(s.z>0.){colour=textureSampleLevel(atlas,linear_sampler,s.xy,0.);}
 output[i]=pack4x8unorm(vec4(colour.rgb,1.));
}
