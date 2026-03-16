#include <gtk/gtk.h>
#include <epoxy/gl.h>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <array>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

struct Vertex {
    float px, py, pz;
    float nx, ny, nz;
    float u, v;
};

struct MeshGpu {
    GLuint vao = 0;
    GLuint vbo = 0;
    GLuint ebo = 0;
    GLsizei index_count = 0;
    int material_index = -1;
};

enum class TextureKind {
    Diffuse,
    Specular,
    Ambient,
    Emissive,
    Height,
    Normals,
    Shininess,
    Opacity,
    Displacement,
    Lightmap,
    Reflection,
    Unknown,
};

struct TextureEntry {
    int id = -1;
    TextureKind kind = TextureKind::Unknown;
    std::string type_name;
    std::string path;
    bool enabled = true;
    GLuint gl_tex = 0;
    std::filesystem::file_time_type mtime{};
};

struct Mat4 {
    std::array<float, 16> m{};
};

static Mat4 mat_identity() {
    Mat4 out{};
    out.m = {1.f, 0.f, 0.f, 0.f,
             0.f, 1.f, 0.f, 0.f,
             0.f, 0.f, 1.f, 0.f,
             0.f, 0.f, 0.f, 1.f};
    return out;
}

static Mat4 mat_mul(const Mat4& a, const Mat4& b) {
    Mat4 out{};
    for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 4; ++r) {
            out.m[c * 4 + r] =
                a.m[0 * 4 + r] * b.m[c * 4 + 0] +
                a.m[1 * 4 + r] * b.m[c * 4 + 1] +
                a.m[2 * 4 + r] * b.m[c * 4 + 2] +
                a.m[3 * 4 + r] * b.m[c * 4 + 3];
        }
    }
    return out;
}

static Mat4 mat_perspective(float fovy_rad, float aspect, float near_z, float far_z) {
    Mat4 out{};
    const float f = 1.0f / std::tan(fovy_rad / 2.0f);
    out.m = {f / aspect, 0.f, 0.f, 0.f,
             0.f, f, 0.f, 0.f,
             0.f, 0.f, (far_z + near_z) / (near_z - far_z), -1.f,
             0.f, 0.f, (2.f * far_z * near_z) / (near_z - far_z), 0.f};
    return out;
}

static Mat4 mat_translate(float x, float y, float z) {
    Mat4 out = mat_identity();
    out.m[12] = x;
    out.m[13] = y;
    out.m[14] = z;
    return out;
}

static Mat4 mat_rotate_y(float a) {
    Mat4 out = mat_identity();
    out.m[0] = std::cos(a);
    out.m[2] = std::sin(a);
    out.m[8] = -std::sin(a);
    out.m[10] = std::cos(a);
    return out;
}

struct App {
    GtkWidget* window = nullptr;
    GtkWidget* gl_area = nullptr;
    GtkWidget* tree = nullptr;
    GtkListStore* store = nullptr;
    GtkWidget* status = nullptr;

    GLuint shader = 0;
    GLint u_mvp = -1;
    GLint u_model = -1;
    GLint u_tex = -1;
    GLint u_use_tex = -1;
    GLuint white_texture = 0;

    std::vector<MeshGpu> meshes;
    std::vector<TextureEntry> textures;
    std::unordered_map<int, int> diffuse_for_material;
    std::filesystem::path model_path;

    float radius = 3.0f;
    float center_x = 0.0f;
    float center_y = 0.0f;
    float center_z = 0.0f;
    float rotate = 0.0f;

    bool gl_ready = false;
};

enum Columns {
    COL_ENABLED = 0,
    COL_TYPE = 1,
    COL_PATH = 2,
    COL_ID = 3,
    N_COLS
};

static TextureKind map_kind(aiTextureType t, const char** name_out) {
    switch (t) {
        case aiTextureType_DIFFUSE: *name_out = "Diffuse"; return TextureKind::Diffuse;
        case aiTextureType_SPECULAR: *name_out = "Specular"; return TextureKind::Specular;
        case aiTextureType_AMBIENT: *name_out = "Ambient"; return TextureKind::Ambient;
        case aiTextureType_EMISSIVE: *name_out = "Emissive"; return TextureKind::Emissive;
        case aiTextureType_HEIGHT: *name_out = "Height"; return TextureKind::Height;
        case aiTextureType_NORMALS: *name_out = "Normal"; return TextureKind::Normals;
        case aiTextureType_SHININESS: *name_out = "Shininess"; return TextureKind::Shininess;
        case aiTextureType_OPACITY: *name_out = "Opacity"; return TextureKind::Opacity;
        case aiTextureType_DISPLACEMENT: *name_out = "Displacement"; return TextureKind::Displacement;
        case aiTextureType_LIGHTMAP: *name_out = "Lightmap"; return TextureKind::Lightmap;
        case aiTextureType_REFLECTION: *name_out = "Reflection"; return TextureKind::Reflection;
        default: *name_out = "Other"; return TextureKind::Unknown;
    }
}

static GLuint compile_shader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = GL_FALSE;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048] = {};
        glGetShaderInfoLog(s, sizeof(log), nullptr, log);
        std::cerr << "Shader compile error: " << log << '\n';
    }
    return s;
}

static GLuint link_program(const char* vs, const char* fs) {
    GLuint v = compile_shader(GL_VERTEX_SHADER, vs);
    GLuint f = compile_shader(GL_FRAGMENT_SHADER, fs);
    GLuint p = glCreateProgram();
    glAttachShader(p, v);
    glAttachShader(p, f);
    glLinkProgram(p);
    GLint ok = GL_FALSE;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048] = {};
        glGetProgramInfoLog(p, sizeof(log), nullptr, log);
        std::cerr << "Program link error: " << log << '\n';
    }
    glDeleteShader(v);
    glDeleteShader(f);
    return p;
}

static void clear_meshes(App* app) {
    for (auto& m : app->meshes) {
        if (m.ebo) glDeleteBuffers(1, &m.ebo);
        if (m.vbo) glDeleteBuffers(1, &m.vbo);
        if (m.vao) glDeleteVertexArrays(1, &m.vao);
    }
    app->meshes.clear();
}

static void clear_textures(App* app) {
    for (auto& t : app->textures) {
        if (t.gl_tex) glDeleteTextures(1, &t.gl_tex);
    }
    app->textures.clear();
}

static bool load_texture_to_gpu(TextureEntry& t) {
    std::cerr << "Loading texture [" << t.type_name << "] " << t.path << '\n';
    GError* err = nullptr;
    GdkPixbuf* pix = gdk_pixbuf_new_from_file(t.path.c_str(), &err);
    if (!pix) {
        if (err) {
            std::cerr << "Failed to load texture " << t.path << ": " << err->message << '\n';
            g_error_free(err);
        }
        return false;
    }

    const int width = gdk_pixbuf_get_width(pix);
    const int height = gdk_pixbuf_get_height(pix);
    const int channels = gdk_pixbuf_get_n_channels(pix);
    const int rowstride = gdk_pixbuf_get_rowstride(pix);
    guchar* pixels = gdk_pixbuf_get_pixels(pix);

    std::vector<unsigned char> tightly_packed(static_cast<size_t>(width * height * channels));
    for (int y = 0; y < height; ++y) {
        memcpy(tightly_packed.data() + static_cast<size_t>(y * width * channels),
               pixels + y * rowstride,
               static_cast<size_t>(width * channels));
    }

    GLenum fmt = (channels == 4) ? GL_RGBA : GL_RGB;
    if (!t.gl_tex) glGenTextures(1, &t.gl_tex);
    glBindTexture(GL_TEXTURE_2D, t.gl_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexImage2D(GL_TEXTURE_2D, 0, fmt, width, height, 0, fmt, GL_UNSIGNED_BYTE, tightly_packed.data());
    glGenerateMipmap(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);

    g_object_unref(pix);

    std::error_code ec;
    t.mtime = std::filesystem::last_write_time(t.path, ec);
    std::cerr << "Loaded texture id=" << t.id
              << " gl_tex=" << t.gl_tex
              << " size=" << width << "x" << height
              << " channels=" << channels << '\n';
    return true;
}

static void refresh_tree(App* app) {
    gtk_list_store_clear(app->store);
    for (const auto& t : app->textures) {
        GtkTreeIter iter;
        gtk_list_store_append(app->store, &iter);
        gtk_list_store_set(app->store, &iter,
                           COL_ENABLED, t.enabled,
                           COL_TYPE, t.type_name.c_str(),
                           COL_PATH, t.path.c_str(),
                           COL_ID, t.id,
                           -1);
    }
}

static void upload_mesh(MeshGpu& mg, const std::vector<Vertex>& v, const std::vector<unsigned int>& idx) {
    glGenVertexArrays(1, &mg.vao);
    glGenBuffers(1, &mg.vbo);
    glGenBuffers(1, &mg.ebo);

    glBindVertexArray(mg.vao);
    glBindBuffer(GL_ARRAY_BUFFER, mg.vbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(v.size() * sizeof(Vertex)), v.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mg.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, static_cast<GLsizeiptr>(idx.size() * sizeof(unsigned int)), idx.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, px)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, nx)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, u)));

    glBindVertexArray(0);
    mg.index_count = static_cast<GLsizei>(idx.size());
}

static bool load_model(App* app, const std::string& filename) {
    std::cerr << "Loading model: " << filename << '\n';
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(
        filename,
        aiProcess_Triangulate |
        aiProcess_PreTransformVertices |
        aiProcess_GenSmoothNormals |
        aiProcess_JoinIdenticalVertices |
        aiProcess_ImproveCacheLocality |
        aiProcess_FlipUVs);

    if (!scene || !scene->HasMeshes() || (scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE)) {
        std::string err = importer.GetErrorString();
        if (err.empty()) err = "unknown Assimp error";
        std::string status = "Failed to load model: " + err;
        gtk_label_set_text(GTK_LABEL(app->status), status.c_str());
        std::cerr << status << '\n';
        return false;
    }

    std::cerr << "Assimp scene loaded: meshes=" << scene->mNumMeshes
              << " materials=" << scene->mNumMaterials
              << " embedded_textures=" << scene->mNumTextures << '\n';

    if (app->gl_ready) {
        clear_meshes(app);
        clear_textures(app);
    } else {
        app->meshes.clear();
        app->textures.clear();
    }
    app->diffuse_for_material.clear();

    app->model_path = filename;
    auto base_dir = std::filesystem::path(filename).parent_path();

    int next_tex_id = 0;
    std::unordered_map<std::string, int> key_to_id;

    auto add_texture = [&](int mat_idx, aiTextureType type, const aiString& path_str) {
        const char* type_name = "Other";
        TextureKind kind = map_kind(type, &type_name);

        std::filesystem::path p(path_str.C_Str());
        if (p.is_relative()) {
            p = base_dir / p;
        }
        p = p.lexically_normal();

        std::string key = std::string(type_name) + "::" + p.string();
        auto it = key_to_id.find(key);
        int tid;
        if (it == key_to_id.end()) {
            TextureEntry t;
            t.id = next_tex_id++;
            t.kind = kind;
            t.type_name = type_name;
            t.path = p.string();
            t.enabled = true;
            app->textures.push_back(t);
            tid = t.id;
            key_to_id.emplace(key, tid);
        } else {
            tid = it->second;
        }

        if (type == aiTextureType_DIFFUSE && !app->diffuse_for_material.count(mat_idx)) {
            app->diffuse_for_material[mat_idx] = tid;
        }
    };

    for (unsigned int i = 0; i < scene->mNumMaterials; ++i) {
        aiMaterial* mat = scene->mMaterials[i];
        const aiTextureType candidates[] = {
            aiTextureType_DIFFUSE,
            aiTextureType_SPECULAR,
            aiTextureType_AMBIENT,
            aiTextureType_EMISSIVE,
            aiTextureType_HEIGHT,
            aiTextureType_NORMALS,
            aiTextureType_LIGHTMAP,
            aiTextureType_OPACITY,
            aiTextureType_DISPLACEMENT,
            aiTextureType_REFLECTION,
        };
        for (aiTextureType type : candidates) {
            unsigned int count = mat->GetTextureCount(type);
            for (unsigned int t = 0; t < count; ++t) {
                aiString path;
                if (mat->GetTexture(type, t, &path) == aiReturn_SUCCESS) {
                    add_texture(static_cast<int>(i), type, path);
                }
            }
        }
    }

    float min_x = INFINITY, min_y = INFINITY, min_z = INFINITY;
    float max_x = -INFINITY, max_y = -INFINITY, max_z = -INFINITY;

    for (unsigned int i = 0; i < scene->mNumMeshes; ++i) {
        aiMesh* mesh = scene->mMeshes[i];
        std::cerr << "Processing mesh[" << i << "] name='" << mesh->mName.C_Str()
                  << "' vertices=" << mesh->mNumVertices
                  << " faces=" << mesh->mNumFaces
                  << " material_index=" << mesh->mMaterialIndex << '\n';
        std::vector<Vertex> vertices;
        vertices.reserve(mesh->mNumVertices);
        for (unsigned int v = 0; v < mesh->mNumVertices; ++v) {
            Vertex vert{};
            vert.px = mesh->mVertices[v].x;
            vert.py = mesh->mVertices[v].y;
            vert.pz = mesh->mVertices[v].z;
            if (mesh->HasNormals()) {
                vert.nx = mesh->mNormals[v].x;
                vert.ny = mesh->mNormals[v].y;
                vert.nz = mesh->mNormals[v].z;
            } else {
                vert.nx = 0.f; vert.ny = 1.f; vert.nz = 0.f;
            }
            if (mesh->HasTextureCoords(0)) {
                vert.u = mesh->mTextureCoords[0][v].x;
                vert.v = mesh->mTextureCoords[0][v].y;
            } else {
                vert.u = vert.v = 0.f;
            }
            vertices.push_back(vert);

            min_x = std::min(min_x, vert.px); min_y = std::min(min_y, vert.py); min_z = std::min(min_z, vert.pz);
            max_x = std::max(max_x, vert.px); max_y = std::max(max_y, vert.py); max_z = std::max(max_z, vert.pz);
        }

        std::vector<unsigned int> indices;
        for (unsigned int f = 0; f < mesh->mNumFaces; ++f) {
            const aiFace& face = mesh->mFaces[f];
            for (unsigned int j = 0; j < face.mNumIndices; ++j) {
                indices.push_back(face.mIndices[j]);
            }
        }

        MeshGpu mg;
        mg.material_index = static_cast<int>(mesh->mMaterialIndex);
        if (app->gl_ready) upload_mesh(mg, vertices, indices);
        app->meshes.push_back(mg);
    }

    app->center_x = (min_x + max_x) * 0.5f;
    app->center_y = (min_y + max_y) * 0.5f;
    app->center_z = (min_z + max_z) * 0.5f;
    const float dx = max_x - min_x;
    const float dy = max_y - min_y;
    const float dz = max_z - min_z;
    app->radius = std::max(0.1f, std::sqrt(dx * dx + dy * dy + dz * dz) * 0.7f);

    if (app->gl_ready) {
        for (auto& t : app->textures) load_texture_to_gpu(t);
    }

    std::cerr << "Model ready: gpu_meshes=" << app->meshes.size()
              << " textures=" << app->textures.size() << '\n';

    refresh_tree(app);
    gtk_label_set_text(GTK_LABEL(app->status), filename.c_str());
    gtk_gl_area_queue_render(GTK_GL_AREA(app->gl_area));
    return true;
}

static void reload_textures(App* app) {
    if (!app->gl_ready) return;
    gtk_gl_area_make_current(GTK_GL_AREA(app->gl_area));
    for (auto& t : app->textures) {
        if (!std::filesystem::exists(t.path)) continue;
        std::error_code ec;
        auto mt = std::filesystem::last_write_time(t.path, ec);
        if (ec) continue;
        if (mt != t.mtime) {
            load_texture_to_gpu(t);
        }
    }
    gtk_gl_area_queue_render(GTK_GL_AREA(app->gl_area));
}

static gboolean on_focus_in(GtkWidget*, GdkEventFocus*, gpointer data) {
    auto* app = static_cast<App*>(data);
    reload_textures(app);
    return FALSE;
}

static gboolean on_render(GtkGLArea* area, GdkGLContext*, gpointer data) {
    auto* app = static_cast<App*>(data);
    int w = gtk_widget_get_allocated_width(GTK_WIDGET(area));
    int h = gtk_widget_get_allocated_height(GTK_WIDGET(area));
    if (w <= 0 || h <= 0) return TRUE;

    app->rotate += 0.005f;
    glViewport(0, 0, w, h);
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.08f, 0.09f, 0.11f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(app->shader);

    Mat4 proj = mat_perspective(60.f * static_cast<float>(M_PI) / 180.f, static_cast<float>(w) / static_cast<float>(h), 0.01f, 500.f);
    Mat4 view = mat_translate(-app->center_x, -app->center_y, -app->center_z - app->radius * 2.2f);
    Mat4 rot = mat_rotate_y(app->rotate);
    Mat4 model = rot;
    Mat4 mvp = mat_mul(proj, mat_mul(view, model));

    glUniformMatrix4fv(app->u_mvp, 1, GL_FALSE, mvp.m.data());
    glUniformMatrix4fv(app->u_model, 1, GL_FALSE, model.m.data());
    glUniform1i(app->u_tex, 0);

    for (const auto& m : app->meshes) {
        GLuint tex = app->white_texture;
        bool use_tex = false;

        auto it = app->diffuse_for_material.find(m.material_index);
        if (it != app->diffuse_for_material.end()) {
            int id = it->second;
            if (id >= 0 && id < static_cast<int>(app->textures.size())) {
                const auto& t = app->textures[id];
                if (t.enabled && t.gl_tex) {
                    tex = t.gl_tex;
                    use_tex = true;
                }
            }
        }

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex);
        glUniform1i(app->u_use_tex, use_tex ? 1 : 0);
        glBindVertexArray(m.vao);
        glDrawElements(GL_TRIANGLES, m.index_count, GL_UNSIGNED_INT, nullptr);
    }

    glBindVertexArray(0);
    glUseProgram(0);
    gtk_gl_area_queue_render(area);
    return TRUE;
}

static void ensure_gpu_resources(App* app) {
    if (app->shader) return;
    const char* vs = R"(
        #version 330 core
        layout(location = 0) in vec3 a_pos;
        layout(location = 1) in vec3 a_norm;
        layout(location = 2) in vec2 a_uv;
        uniform mat4 u_mvp;
        uniform mat4 u_model;
        out vec3 v_n;
        out vec2 v_uv;
        void main() {
            gl_Position = u_mvp * vec4(a_pos, 1.0);
            v_n = mat3(u_model) * a_norm;
            v_uv = a_uv;
        }
    )";

    const char* fs = R"(
        #version 330 core
        in vec3 v_n;
        in vec2 v_uv;
        uniform sampler2D u_tex;
        uniform int u_use_tex;
        out vec4 frag;
        void main() {
            vec3 n = normalize(v_n);
            vec3 l = normalize(vec3(0.3, 0.8, 0.4));
            float ndl = max(dot(n, l), 0.12);
            vec3 base = (u_use_tex == 1) ? texture(u_tex, v_uv).rgb : vec3(0.85);
            frag = vec4(base * ndl, 1.0);
        }
    )";

    app->shader = link_program(vs, fs);
    app->u_mvp = glGetUniformLocation(app->shader, "u_mvp");
    app->u_model = glGetUniformLocation(app->shader, "u_model");
    app->u_tex = glGetUniformLocation(app->shader, "u_tex");
    app->u_use_tex = glGetUniformLocation(app->shader, "u_use_tex");

    glGenTextures(1, &app->white_texture);
    glBindTexture(GL_TEXTURE_2D, app->white_texture);
    unsigned char white[4] = {255, 255, 255, 255};
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, white);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    for (auto& mesh : app->meshes) {
        if (mesh.vao == 0) {
            // No CPU-side mesh cache retained, so this path is for first load after realize.
        }
    }

    for (auto& t : app->textures) load_texture_to_gpu(t);
}

static void on_realize(GtkGLArea* area, gpointer data) {
    auto* app = static_cast<App*>(data);
    gtk_gl_area_make_current(area);
    if (gtk_gl_area_get_error(area)) return;
    app->gl_ready = true;
    ensure_gpu_resources(app);

    if (!app->model_path.empty()) {
        load_model(app, app->model_path.string());
    }
}

static void on_unrealize(GtkGLArea* area, gpointer data) {
    auto* app = static_cast<App*>(data);
    gtk_gl_area_make_current(area);
    clear_meshes(app);
    clear_textures(app);
    if (app->white_texture) glDeleteTextures(1, &app->white_texture);
    if (app->shader) glDeleteProgram(app->shader);
    app->shader = 0;
    app->white_texture = 0;
    app->gl_ready = false;
}

static void on_toggle(GtkCellRendererToggle* cell, gchar* path, gpointer data) {
    auto* app = static_cast<App*>(data);
    GtkTreePath* tp = gtk_tree_path_new_from_string(path);
    GtkTreeIter iter;
    if (!gtk_tree_model_get_iter(GTK_TREE_MODEL(app->store), &iter, tp)) {
        gtk_tree_path_free(tp);
        return;
    }

    gboolean enabled;
    int id;
    gtk_tree_model_get(GTK_TREE_MODEL(app->store), &iter, COL_ENABLED, &enabled, COL_ID, &id, -1);
    enabled = !enabled;
    gtk_list_store_set(app->store, &iter, COL_ENABLED, enabled, -1);

    if (id >= 0 && id < static_cast<int>(app->textures.size())) {
        app->textures[id].enabled = enabled;
    }

    gtk_tree_path_free(tp);
    gtk_gl_area_queue_render(GTK_GL_AREA(app->gl_area));
    (void)cell;
}

static void on_open_clicked(GtkButton*, gpointer data) {
    auto* app = static_cast<App*>(data);
    GtkWidget* dialog = gtk_file_chooser_dialog_new(
        "Open Model",
        GTK_WINDOW(app->window),
        GTK_FILE_CHOOSER_ACTION_OPEN,
        "_Cancel", GTK_RESPONSE_CANCEL,
        "_Open", GTK_RESPONSE_ACCEPT,
        nullptr);

    if (gtk_dialog_run(GTK_DIALOG(dialog)) == GTK_RESPONSE_ACCEPT) {
        char* filename = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dialog));
        if (filename) {
            load_model(app, filename);
            g_free(filename);
        }
    }
    gtk_widget_destroy(dialog);
}

static GtkWidget* build_ui(App* app) {
    app->window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_default_size(GTK_WINDOW(app->window), 1400, 900);
    gtk_window_set_title(GTK_WINDOW(app->window), "3D Texture Viewer");

    GtkWidget* root = gtk_box_new(GTK_ORIENTATION_VERTICAL, 6);
    gtk_container_add(GTK_CONTAINER(app->window), root);

    GtkWidget* toolbar = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);
    GtkWidget* open_btn = gtk_button_new_with_label("Open Model");
    gtk_box_pack_start(GTK_BOX(toolbar), open_btn, FALSE, FALSE, 0);

    app->status = gtk_label_new("No model loaded");
    gtk_label_set_xalign(GTK_LABEL(app->status), 0.0f);
    gtk_box_pack_start(GTK_BOX(toolbar), app->status, TRUE, TRUE, 0);

    gtk_box_pack_start(GTK_BOX(root), toolbar, FALSE, FALSE, 0);

    GtkWidget* paned = gtk_paned_new(GTK_ORIENTATION_HORIZONTAL);
    gtk_box_pack_start(GTK_BOX(root), paned, TRUE, TRUE, 0);

    app->gl_area = gtk_gl_area_new();
    gtk_gl_area_set_has_depth_buffer(GTK_GL_AREA(app->gl_area), TRUE);
    gtk_gl_area_set_has_alpha(GTK_GL_AREA(app->gl_area), FALSE);
    gtk_widget_set_hexpand(app->gl_area, TRUE);
    gtk_widget_set_vexpand(app->gl_area, TRUE);
    gtk_paned_pack1(GTK_PANED(paned), app->gl_area, TRUE, FALSE);

    app->store = gtk_list_store_new(N_COLS, G_TYPE_BOOLEAN, G_TYPE_STRING, G_TYPE_STRING, G_TYPE_INT);
    app->tree = gtk_tree_view_new_with_model(GTK_TREE_MODEL(app->store));

    GtkCellRenderer* toggle = gtk_cell_renderer_toggle_new();
    g_signal_connect(toggle, "toggled", G_CALLBACK(on_toggle), app);
    GtkTreeViewColumn* col_toggle = gtk_tree_view_column_new_with_attributes("On", toggle, "active", COL_ENABLED, nullptr);
    gtk_tree_view_append_column(GTK_TREE_VIEW(app->tree), col_toggle);

    GtkCellRenderer* text = gtk_cell_renderer_text_new();
    GtkTreeViewColumn* col_type = gtk_tree_view_column_new_with_attributes("Type", text, "text", COL_TYPE, nullptr);
    gtk_tree_view_append_column(GTK_TREE_VIEW(app->tree), col_type);

    GtkCellRenderer* text2 = gtk_cell_renderer_text_new();
    GtkTreeViewColumn* col_path = gtk_tree_view_column_new_with_attributes("Texture Path", text2, "text", COL_PATH, nullptr);
    gtk_tree_view_append_column(GTK_TREE_VIEW(app->tree), col_path);

    GtkWidget* scrolled = gtk_scrolled_window_new(nullptr, nullptr);
    gtk_container_add(GTK_CONTAINER(scrolled), app->tree);
    gtk_widget_set_size_request(scrolled, 400, -1);
    gtk_paned_pack2(GTK_PANED(paned), scrolled, FALSE, FALSE);

    g_signal_connect(open_btn, "clicked", G_CALLBACK(on_open_clicked), app);
    g_signal_connect(app->window, "focus-in-event", G_CALLBACK(on_focus_in), app);
    g_signal_connect(app->window, "destroy", G_CALLBACK(gtk_main_quit), nullptr);
    g_signal_connect(app->gl_area, "realize", G_CALLBACK(on_realize), app);
    g_signal_connect(app->gl_area, "unrealize", G_CALLBACK(on_unrealize), app);
    g_signal_connect(app->gl_area, "render", G_CALLBACK(on_render), app);

    return app->window;
}

int main(int argc, char** argv) {
    gtk_init(&argc, &argv);

    App app;
    GtkWidget* w = build_ui(&app);
    gtk_widget_show_all(w);

    if (argc > 1) {
        load_model(&app, argv[1]);
    }

    gtk_main();
    return 0;
}
