
first

- in the loaded view, the widowx model just looks like dots
- STL mesh loading into WASM VFS and `from_xml_path` interaction with `<include>` and `meshdir` should work but may need debugging

later

- the targets look like thick cylinders
- free camera orientation looks wrong (table is sideways)
- The `MjvGeom` property access patterns (`.pos`, `.mat`, `.rgba`, `.size`) may return embind wrapper objects rather than raw TypedArrays — if so, you'd need to use getter methods
- The Gemini model name is set to `gemini-2.0-flash` (general availability) rather than `gemini-robotics-er-1.5-preview` (which may require allowlisting) — you may want to switch this based on your API key's access
