
gemini web
- The `@mujoco/mujoco` WASM module import from CDN may need CORS adjustments or a bundler if the dynamic import doesn't resolve the `.wasm` file correctly
- The `MjvGeom` property access patterns (`.pos`, `.mat`, `.rgba`, `.size`) may return embind wrapper objects rather than raw TypedArrays — if so, you'd need to use getter methods
- STL mesh loading into WASM VFS and `from_xml_path` interaction with `<include>` and `meshdir` should work but may need debugging
- The Gemini model name is set to `gemini-2.0-flash` (general availability) rather than `gemini-robotics-er-1.5-preview` (which may require allowlisting) — you may want to switch this based on your API key's access
