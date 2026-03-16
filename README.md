# 3dtexview

GTK3 + Assimp desktop viewer for 3D meshes with live texture reloading.

## Features

- Load common 3D model formats through **Assimp** (`obj`, `fbx`, `dae`, `3ds`, etc.).
- OpenGL viewport on the left (`GtkGLArea`) to display the model.
- Texture list on the right (`GtkTreeView`) with per-texture checkboxes.
- Texture categories include diffuse/specular/normal/lightmap/shadow-adjacent map types that Assimp reports.
- When the viewer window regains focus (for example after editing in GIMP), textures are checked for file timestamp changes and reloaded automatically.

## Build

```bash
cmake -S . -B build
cmake --build build
```

Dependencies:

- `gtk+-3.0`
- `assimp`
- `epoxy`
- C++17 compiler

## Run

```bash
./build/3dtexview /path/to/model.obj
```

Or run without arguments and click **Open Model**.
