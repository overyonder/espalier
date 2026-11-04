fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let source = std::fs::read_to_string(&args[1])?;
    let module = naga::front::wgsl::parse_str(&source).map_err(|e| e.emit_to_string(&source))?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)?;
    let pipeline = naga::back::spv::PipelineOptions {
        shader_stage: naga::ShaderStage::Compute,
        entry_point: args.get(3).map_or("compress_bc7", String::as_str).into(),
    };
    let mut options = naga::back::spv::Options::default();
    if pipeline.entry_point == "prefix" {
        options.lang_version = (1, 3);
    }
    let words = naga::back::spv::write_vec(&module, &info, &options, Some(&pipeline))?;
    std::fs::write(
        &args[2],
        words
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    Ok(())
}
