@group(2) @binding(4) var<uniform> inverse_cameras:array<mat4x4<f32>,16>;

// Compare both depths at the raster sample's centre. Comparing the projected
// texel against a different point on a sloping surface rejects visible texels.
fn visibility_depth(point:vec3<f32>, normal:vec3<f32>, camera:i32,
                    ndc:vec3<f32>, pixel:vec2<i32>, dimensions:vec2<i32>)->f32 {
 let plane=transpose(inverse_cameras[camera])*vec4(normal,-dot(normal,point));
 let sample_xy=(2.*(vec2<f32>(pixel)+.5)/vec2<f32>(dimensions)-1.)*vec2(1.,-1.);
 let correction=dot(plane.xy,sample_xy-ndc.xy)/plane.z;
 return (ndc.z-correction)*.5+.5;
}
