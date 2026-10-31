@group(0) @binding(0) var<storage,read> counts:array<u32>;
@group(0) @binding(1) var<storage,read_write> offsets:array<u32>;
var<workgroup> subgroup_totals:array<u32,256>;
@compute @workgroup_size(256)
fn prefix(@builtin(local_invocation_index) lane:u32,
          @builtin(subgroup_id) group:u32,
          @builtin(subgroup_invocation_id) subgroup_lane:u32,
          @builtin(num_subgroups) group_count:u32){
 let first=lane*64u;var sum=0u;
 for(var i=0u;i<64u;i++){sum+=counts[first+i];}
 let local_offset=subgroupExclusiveAdd(sum);let group_sum=subgroupAdd(sum);
 if(subgroup_lane==0u){subgroup_totals[group]=group_sum;}
 workgroupBarrier();
 if(group==0u){
  var group_value=0u;if(subgroup_lane<group_count){group_value=subgroup_totals[subgroup_lane];}
  let group_offset=subgroupExclusiveAdd(group_value);
  if(subgroup_lane<group_count){subgroup_totals[subgroup_lane]=group_offset;}
 }
 workgroupBarrier();
 var offset=subgroup_totals[group]+local_offset;
 for(var i=0u;i<64u;i++){offsets[first+i]=offset;offset+=counts[first+i];}
 if(lane==255u){offsets[16384u]=offset;}
}
