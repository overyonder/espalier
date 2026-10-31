// UV bins hold triangle IDs; colours and surface attributes stay in registers.
@group(1) @binding(0) var<storage,read> positions:array<f32>;
@group(1) @binding(1) var<storage,read> normals:array<f32>;
@group(1) @binding(2) var<storage,read> uv_data:array<f32>;
@group(1) @binding(3) var<storage,read> index_data:array<u32>;
@group(1) @binding(4) var<storage,read> offsets:array<u32>;
@group(1) @binding(5) var<storage,read> triangles:array<u32>;
struct Cameras { matrices:array<mat4x4<f32>,13>, origins:array<vec4<f32>,13>, metadata:vec4<u32> }
@group(2) @binding(0) var paintings:texture_2d_array<f32>;
@group(2) @binding(1) var depths:texture_depth_2d_array;
@group(2) @binding(2) var linear_sampler:sampler;
@group(2) @binding(3) var<uniform> cameras:Cameras;
fn vertex_index(i:u32)->u32 {
 if(cameras.metadata.z==2u){return (index_data[i/2u]>>((i%2u)*16u))&65535u;}
 return index_data[i];
}
fn vertex_uv(i:u32)->vec2<f32>{return vec2(uv_data[i*2u],uv_data[i*2u+1u])*4096.;}
fn vertex_position(i:u32)->vec3<f32>{return vec3(positions[i*3u],-positions[i*3u+2u],positions[i*3u+1u]);}
fn vertex_normal(i:u32)->vec3<f32>{return vec3(normals[i*3u],-normals[i*3u+2u],normals[i*3u+1u]);}
fn to_linear(c:vec3<f32>)->vec3<f32>{return select(c/12.92,pow((c+.055)/1.055,vec3(2.4)),c>=vec3(.04045));}
fn to_srgb(c:vec3<f32>)->vec3<f32>{return select(c*12.92,1.055*pow(max(c,vec3(0)),vec3(1./2.4))-.055,c>=vec3(.0031308));}
fn project_surface(point:vec3<f32>,normal:vec3<f32>)->vec3<f32>{
 var total=vec3(0.);var weights=0.;let n=normalize(normal);
 for(var i=0;i<13;i++){
  let cosine=dot(n,normalize(cameras.origins[i].xyz-point));if(cosine<.17364818){continue;}
  let q=cameras.matrices[i]*vec4(point,1.);if(q.w<=0.){continue;}
  let ndc=q.xyz/q.w;let uv=ndc.xy*.5+.5;
  if(any(uv<vec2(0.))||any(uv>vec2(1.))){continue;}
  let dimensions=vec2<i32>(textureDimensions(depths));
  let pixel=clamp(vec2<i32>(vec2(uv.x,1.-uv.y)*vec2<f32>(dimensions)),vec2(0),dimensions-1);
  let stored=textureLoad(depths,pixel,i,0);let z=ndc.z*.5+.5;
  let distance=.01*1000./(1000.-z*(1000.-.01));let seen=.01*1000./(1000.-stored*(1000.-.01));
  if(abs(distance-seen)>.0015){continue;}
  let edge=clamp(min(min(uv.x,uv.y),min(1.-uv.x,1.-uv.y))/.08,0.,1.);
  let weight=pow(cosine,10.)*edge;
  let input_uv=uv*f32(cameras.metadata.x)/f32(cameras.metadata.y);
  total+=to_linear(textureSampleLevel(paintings,linear_sampler,input_uv,i,0.).rgb)*weight;weights+=weight;
 }
 if(weights>1.e-12){return to_srgb(total/weights);}return vec3(0.);
}
fn project_pixel(coordinate:vec2<u32>)->vec4<f32>{
 let p=vec2<f32>(coordinate)+.5;let tile=min(coordinate/32u,vec2(127u));let bin=tile.y*128u+tile.x;
 var inside=false;var best_face=0xffffffffu;
 var best_distance=16.0001;var best_bary=vec3(0.);var best_vertices=vec3(0u);var found=false;
 for(var k=offsets[bin];k<offsets[bin+1u];k++){
  let face=triangles[k];let vertices=vec3(vertex_index(face*3u),vertex_index(face*3u+1u),vertex_index(face*3u+2u));
  let a=vertex_uv(vertices.x);let b=vertex_uv(vertices.y);let c=vertex_uv(vertices.z);let ab=b-a;let ac=c-a;let ap=p-a;
  let determinant=ab.x*ac.y-ab.y*ac.x;if(abs(determinant)<1.e-14){continue;}
  let v=(ap.x*ac.y-ap.y*ac.x)/determinant;let w=(ab.x*ap.y-ab.y*ap.x)/determinant;let bary=vec3(1.-v-w,v,w);
  if(all(bary>=vec3(-1.e-6))){
   if(!inside || face<best_face){best_vertices=vertices;best_bary=bary;best_face=face;found=true;}
   inside=true;continue;
  }
  if(inside){continue;}
  let ends=array<vec2<f32>,3>(a,b,c);let corners=array<vec3<f32>,3>(vec3(1.,0.,0.),vec3(0.,1.,0.),vec3(0.,0.,1.));
  for(var edge=0u;edge<3u;edge++){
   let next=(edge+1u)%3u;let delta=ends[next]-ends[edge];let t=clamp(dot(p-ends[edge],delta)/max(dot(delta,delta),1.e-20),0.,1.);
   let d=p-(ends[edge]+t*delta);let distance=dot(d,d);
   if(distance<best_distance || (distance==best_distance && face<best_face)){best_distance=distance;best_bary=mix(corners[edge],corners[next],t);best_vertices=vertices;best_face=face;found=true;}
  }
 }
 if(!found){return vec4(0.,0.,0.,1.);}
 let point=vertex_position(best_vertices.x)*best_bary.x+vertex_position(best_vertices.y)*best_bary.y+vertex_position(best_vertices.z)*best_bary.z;
 var normal:vec3<f32>;
 if(cameras.metadata.w==0u){
  normal=vertex_normal(best_vertices.x)*best_bary.x+vertex_normal(best_vertices.y)*best_bary.y+vertex_normal(best_vertices.z)*best_bary.z;
 }else{
  // glTF primitives without NORMAL have implicit flat triangle normals.
  normal=cross(vertex_position(best_vertices.y)-vertex_position(best_vertices.x),vertex_position(best_vertices.z)-vertex_position(best_vertices.x));
 }
 return vec4(floor(clamp(project_surface(point,normal),vec3(0.),vec3(1.))*255.+.5)/255.,1.);
}
