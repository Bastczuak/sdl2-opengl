#[allow(clippy::all)]
mod gl {
  include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

use freetype as ft;
use gl::types::*;
use image::GenericImageView;
use lyon::{
  math::{rect, Point},
  tessellation::{
    geometry_builder::simple_builder, BuffersBuilder, StrokeOptions, StrokeTessellator, StrokeVertex,
    StrokeVertexConstructor, VertexBuffers,
  },
};
use sdl2::{
  event::{Event, WindowEvent},
  keyboard::Keycode,
  video::GLProfile,
};
use std::{ffi::CString, ops::Deref, path::Path};

macro_rules! get_offset {
  ($type:ty, $field:tt) => {{
    let dummy = core::mem::MaybeUninit::<$type>::uninit();
    let dummy_ptr = dummy.as_ptr();
    let field_ptr = core::ptr::addr_of!((*dummy_ptr).$field);
    field_ptr as usize - dummy_ptr as usize
  }};
}
macro_rules! cstr {
  ($literal:expr) => {
    (std::ffi::CStr::from_bytes_with_nul_unchecked(concat!($literal, "\0").as_bytes()).as_ptr())
  };
}

pub struct Gl {
  inner: std::rc::Rc<gl::Gl>,
}

impl Gl {
  fn load_with<F>(load_fn: F) -> Self
  where
    F: FnMut(&'static str) -> *const GLvoid,
  {
    Self {
      inner: std::rc::Rc::new(gl::Gl::load_with(load_fn)),
    }
  }
}

impl Deref for Gl {
  type Target = gl::Gl;

  fn deref(&self) -> &Self::Target {
    &self.inner
  }
}

const CUBE_VERTICES: [f32; 16] = [
  // positions //texture coords
  0.5, 0.5, 1.0, 1.0, 0.5, -0.5, 1.0, 0.0, -0.5, -0.5, 0.0, 0.0, -0.5, 0.5, 0.0, 1.0,
];

const CUBE_INDICES: [u16; 6] = [0, 1, 3, 1, 2, 3];

const CUBE_POSITIONS: [(f32, f32, f32); 10] = [
  (0.0, 0.0, 0.0),
  (2.0, 5.0, -15.0),
  (-1.5, -2.2, -2.5),
  (-3.8, -2.0, -12.3),
  (2.4, -0.4, -3.5),
  (-1.7, 3.0, -7.5),
  (1.3, -2.0, -2.5),
  (1.5, 2.0, -2.5),
  (1.5, 0.2, -1.5),
  (-1.3, 1.0, -1.5),
];

const VERTEX_SHADER: &str = r#"
#version 330 core

layout (location = 0) in vec2 Position;
layout (location = 1) in vec2 TexCoords;

uniform mat4 uMVP;

out VERTEX_SHADER_OUTPUT {
  vec2 TexCoords;
} OUT;

void main() {
  gl_Position = uMVP * vec4(Position, 0.0, 1.0);
  OUT.TexCoords = TexCoords;
}
"#;
const FRAGMENT_SHADER: &str = r#"
#version 330 core

in VERTEX_SHADER_OUTPUT {
  vec2 TexCoords;
} IN;

out vec4 Color;

uniform sampler2D uTexture;

void main() {
  Color = texture(uTexture, IN.TexCoords);
}
"#;

const LYON_VERTEX_SHADER: &str = r#"
#version 330 core

layout (location = 0) in mat4 Transform;
layout (location = 4) in vec4 Color;
layout (location = 5) in vec2 Position;

uniform mat4 uMVP;

out VERTEX_SHADER_OUTPUT {
  vec4 Color;
} OUT;

void main() {
  gl_Position = uMVP * Transform * vec4(Position, 0.0, 1.0);
  OUT.Color = Color;
}
"#;
const LYON_FRAGMENT_SHADER: &str = r#"
#version 330 core

in VERTEX_SHADER_OUTPUT {
  vec4 Color;
} IN;

out vec4 Color;

void main() {
  Color = IN.Color;
}
"#;

const FBO_VERTEX_SHADER: &str = r#"
#version 330 core

layout (location = 0) in vec2 Position;
layout (location = 1) in vec2 TexCoords;

out VERTEX_SHADER_OUTPUT {
  vec2 TexCoords;
} OUT;

void main() {
  OUT.TexCoords = TexCoords;
  gl_Position = vec4(Position, 0.0, 1.0);
}
"#;

const FBO_FRAGMENT_SHADER: &str = r#"
#version 330 core

in VERTEX_SHADER_OUTPUT {
  vec2 TexCoords;
} IN;

out vec4 Color;

uniform sampler2D uTexture;

void main() {
  Color = texture(uTexture, IN.TexCoords);
}
"#;

const TEXT_VERTEX_SHADER: &str = r#"
#version 330 core

layout (location = 0) in vec4 PosTex;
layout (location = 1) in vec4 Color;

uniform mat4 uProjection;

out VERTEX_SHADER_OUTPUT {
  vec2 TexCoords;
  vec4 Color;
} OUT;

void main() {
  gl_Position = uProjection * vec4(PosTex.xy, 0.0, 1.0);
  OUT.TexCoords = PosTex.zw;
  OUT.Color = Color;
}
"#;
const TEXT_FRAGMENT_SHADER: &str = r#"
#version 330 core

in VERTEX_SHADER_OUTPUT {
  vec2 TexCoords;
  vec4 Color;
} IN;

out vec4 Color;

uniform sampler2D uTexture;

void main() {
  vec4 sampled = vec4(1.0, 1.0, 1.0, texture(uTexture, IN.TexCoords).r);
  Color = IN.Color * sampled;
}
"#;

#[repr(C)]
#[derive(Debug)]
pub struct MyTextVertex {
  pos_tex: [f32; 4],
  color_rgba: [f32; 4],
}

struct Character {
  tx: f32,
  tx_1: f32,
  ty: f32,
  width: f32,
  height: f32,
  bearing: glam::Vec2,
  advance: f32,
}

#[repr(C)]
struct MyVertex {
  transform_mat4_1: [f32; 4],
  transform_mat4_2: [f32; 4],
  transform_mat4_3: [f32; 4],
  transform_mat4_4: [f32; 4],
  color_rgba: [f32; 4],
  position: [f32; 2],
}

struct MyVertexConfig {
  transform: glam::Mat4,
  color_rgba: glam::Vec4,
}

impl StrokeVertexConstructor<MyVertex> for MyVertexConfig {
  fn new_vertex(&mut self, vertex: StrokeVertex) -> MyVertex {
    let t = self.transform.to_cols_array_2d();
    MyVertex {
      position: vertex.position().to_array(),
      color_rgba: self.color_rgba.to_array(),
      transform_mat4_1: t[0],
      transform_mat4_2: t[1],
      transform_mat4_3: t[2],
      transform_mat4_4: t[3],
    }
  }
}

unsafe fn create_error_buffer(length: usize) -> CString {
  let mut buffer = Vec::with_capacity(length + 1);
  buffer.extend([b' '].iter().cycle().take(length));
  CString::from_vec_unchecked(buffer)
}

fn compile_shader(gl: &gl::Gl, src: &str, kind: GLenum) -> Result<GLuint, String> {
  unsafe {
    let shader = gl.CreateShader(kind);
    let c_str_src = CString::new(src.as_bytes()).unwrap();
    gl.ShaderSource(shader, 1, &c_str_src.as_ptr(), std::ptr::null());
    gl.CompileShader(shader);
    let mut success = gl::TRUE as GLint;
    gl.GetShaderiv(shader, gl::COMPILE_STATUS, &mut success);

    if success == (gl::FALSE as GLint) {
      let mut len = 0;
      gl.GetShaderiv(shader, gl::INFO_LOG_LENGTH, &mut len);
      let error = create_error_buffer(len as usize);
      gl.GetShaderInfoLog(shader, len, std::ptr::null_mut(), error.as_ptr() as *mut GLchar);
      return Err(error.to_string_lossy().into_owned());
    }
    Ok(shader)
  }
}

unsafe fn load_texture(gl: &gl::Gl, path: &str) -> Result<GLuint, String> {
  let mut texture = 0;
  gl.GenTextures(1, &mut texture);

  let image = image::open(path).map_err(|e| e.to_string())?;
  let (width, height) = image.dimensions();
  let format = if image.color().channel_count() == 3 {
    gl::RGB
  } else {
    gl::RGBA
  };
  gl.BindTexture(gl::TEXTURE_2D, texture);
  gl.TexImage2D(
    gl::TEXTURE_2D,
    0,
    format as i32,
    width as i32,
    height as i32,
    0,
    format,
    gl::UNSIGNED_BYTE,
    image.as_bytes().as_ptr() as *const GLvoid,
  );
  gl.GenerateMipmap(gl::TEXTURE_2D);

  gl.TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::REPEAT as i32);
  gl.TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::REPEAT as i32);
  gl.TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::LINEAR_MIPMAP_LINEAR as i32);
  gl.TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::LINEAR as i32);

  gl.BindTexture(gl::TEXTURE_2D, 0);

  Ok(texture)
}

fn link_program(gl: &gl::Gl, vertex_shader: GLuint, fragment_shader: GLuint) -> Result<GLuint, String> {
  unsafe {
    let program = gl.CreateProgram();
    gl.AttachShader(program, vertex_shader);
    gl.AttachShader(program, fragment_shader);
    gl.LinkProgram(program);
    let mut success = gl::TRUE as GLint;
    gl.GetProgramiv(program, gl::LINK_STATUS, &mut success);

    if success == (gl::FALSE as GLint) {
      let mut len = 0;
      gl.GetProgramiv(program, gl::INFO_LOG_LENGTH, &mut len);
      let error = create_error_buffer(len as usize);
      gl.GetProgramInfoLog(program, len, std::ptr::null_mut(), error.as_ptr() as *mut GLchar);
      return Err(error.to_string_lossy().into_owned());
    }

    gl.DeleteShader(vertex_shader);
    gl.DeleteShader(fragment_shader);

    Ok(program)
  }
}

fn main() -> Result<(), String> {
  let quad_vertices = [
    -1.0f32, 1.0, 0.0, 1.0, -1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, 1.0, 1.0, -1.0, 1.0, 0.0, 1.0,
    1.0, 1.0, 1.0,
  ];

  let sdl_context = sdl2::init().unwrap();
  let video_subsystem = sdl_context.video().unwrap();

  let gl_attr = video_subsystem.gl_attr();
  gl_attr.set_context_profile(GLProfile::Core);
  gl_attr.set_context_version(3, 3);

  let window = video_subsystem
    .window("Window", 800, 600)
    .opengl()
    .resizable()
    .position_centered()
    .build()
    .unwrap();

  let _ctx = window.gl_create_context().unwrap();
  let gl = Gl::load_with(|name| video_subsystem.gl_get_proc_address(name) as *const _);

  debug_assert_eq!(gl_attr.context_profile(), GLProfile::Core);
  debug_assert_eq!(gl_attr.context_version(), (3, 3));

  let path: &Path = Path::new("m5x7.ttf");
  let library = ft::Library::init().unwrap();
  let face = library.new_face(path, 0).unwrap();
  face.set_pixel_sizes(0, 48).unwrap();

  let cube_program = {
    let vertex_shader = compile_shader(&gl, VERTEX_SHADER, gl::VERTEX_SHADER)?;
    let fragment_shader = compile_shader(&gl, FRAGMENT_SHADER, gl::FRAGMENT_SHADER)?;
    link_program(&gl, vertex_shader, fragment_shader)?
  };
  let lyon_program = {
    let vertex_shader = compile_shader(&gl, LYON_VERTEX_SHADER, gl::VERTEX_SHADER)?;
    let fragment_shader = compile_shader(&gl, LYON_FRAGMENT_SHADER, gl::FRAGMENT_SHADER)?;
    link_program(&gl, vertex_shader, fragment_shader)?
  };
  let screen_program = {
    let vertex_shader = compile_shader(&gl, FBO_VERTEX_SHADER, gl::VERTEX_SHADER)?;
    let fragment_shader = compile_shader(&gl, FBO_FRAGMENT_SHADER, gl::FRAGMENT_SHADER)?;
    link_program(&gl, vertex_shader, fragment_shader)?
  };
  let text_program = {
    let vertex_shader = compile_shader(&gl, TEXT_VERTEX_SHADER, gl::VERTEX_SHADER)?;
    let fragment_shader = compile_shader(&gl, TEXT_FRAGMENT_SHADER, gl::FRAGMENT_SHADER)?;
    link_program(&gl, vertex_shader, fragment_shader)?
  };

  let (lyon_vao, lyon_vbo, lyon_ebo) = unsafe {
    let mut geometry: VertexBuffers<Point, u16> = VertexBuffers::new();
    let mut vertex_builder = simple_builder(&mut geometry);
    let mut tessellator = StrokeTessellator::new();
    let mut options = StrokeOptions::default();
    options.line_width = 0.1;
    tessellator
      .tessellate_rectangle(&rect(0.0, 0.0, 1.0, 1.0), &options, &mut vertex_builder)
      .unwrap();
    let (mut vao, mut vbo, mut ebo) = (0, 0, 0);
    gl.GenVertexArrays(1, &mut vao);

    gl.GenBuffers(1, &mut vbo);
    gl.GenBuffers(1, &mut ebo);
    gl.BindVertexArray(vao);
    gl.BindBuffer(gl::ARRAY_BUFFER, vbo);
    gl.BufferData(
      gl::ARRAY_BUFFER,
      (std::mem::size_of::<MyVertex>() * geometry.vertices.len() * 10000) as GLsizeiptr,
      std::ptr::null(),
      gl::DYNAMIC_DRAW,
    );
    gl.BindBuffer(gl::ELEMENT_ARRAY_BUFFER, ebo);
    gl.BufferData(
      gl::ELEMENT_ARRAY_BUFFER,
      (std::mem::size_of::<u16>() * geometry.indices.len() * 10000) as GLsizeiptr,
      std::ptr::null(),
      gl::DYNAMIC_DRAW,
    );

    let transform_attr = gl.GetAttribLocation(lyon_program, cstr!("Transform")) as GLuint;
    gl.EnableVertexAttribArray(transform_attr);
    gl.VertexAttribPointer(
      transform_attr,
      4,
      gl::FLOAT,
      gl::FALSE,
      (std::mem::size_of::<MyVertex>()) as i32,
      get_offset!(MyVertex, transform_mat4_1) as *const GLvoid,
    );
    gl.EnableVertexAttribArray(transform_attr + 1);
    gl.VertexAttribPointer(
      transform_attr + 1,
      4,
      gl::FLOAT,
      gl::FALSE,
      (std::mem::size_of::<MyVertex>()) as i32,
      get_offset!(MyVertex, transform_mat4_2) as *const GLvoid,
    );
    gl.EnableVertexAttribArray(transform_attr + 2);
    gl.VertexAttribPointer(
      transform_attr + 2,
      4,
      gl::FLOAT,
      gl::FALSE,
      (std::mem::size_of::<MyVertex>()) as i32,
      get_offset!(MyVertex, transform_mat4_3) as *const GLvoid,
    );
    gl.EnableVertexAttribArray(transform_attr + 3);
    gl.VertexAttribPointer(
      transform_attr + 3,
      4,
      gl::FLOAT,
      gl::FALSE,
      (std::mem::size_of::<MyVertex>()) as i32,
      get_offset!(MyVertex, transform_mat4_4) as *const GLvoid,
    );
    let color_attr = gl.GetAttribLocation(lyon_program, cstr!("Color"));
    gl.EnableVertexAttribArray(color_attr as u32);
    gl.VertexAttribPointer(
      color_attr as u32,
      4,
      gl::FLOAT,
      gl::FALSE,
      (std::mem::size_of::<MyVertex>()) as i32,
      get_offset!(MyVertex, color_rgba) as *const GLvoid,
    );

    let pos_attr = gl.GetAttribLocation(lyon_program, cstr!("Position"));
    gl.EnableVertexAttribArray(pos_attr as u32);
    gl.VertexAttribPointer(
      pos_attr as u32,
      2,
      gl::FLOAT,
      gl::FALSE,
      (std::mem::size_of::<MyVertex>()) as i32,
      get_offset!(MyVertex, position) as *const GLvoid,
    );

    (vao, vbo, ebo)
  };

  let (cube_vao, cube_vbo, cube_ebo) = unsafe {
    let (mut vao, mut vbo, mut ebo) = (0, 0, 0);
    gl.GenVertexArrays(1, &mut vao);
    gl.GenBuffers(1, &mut vbo);
    gl.GenBuffers(1, &mut ebo);
    gl.BindVertexArray(vao);
    gl.BindBuffer(gl::ARRAY_BUFFER, vbo);
    gl.BufferData(
      gl::ARRAY_BUFFER,
      (CUBE_VERTICES.len() * std::mem::size_of::<f32>()) as GLsizeiptr,
      CUBE_VERTICES.as_ptr() as *const GLvoid,
      gl::STATIC_DRAW,
    );
    gl.BindBuffer(gl::ELEMENT_ARRAY_BUFFER, ebo);
    gl.BufferData(
      gl::ELEMENT_ARRAY_BUFFER,
      (CUBE_INDICES.len() * std::mem::size_of::<u16>()) as GLsizeiptr,
      CUBE_INDICES.as_ptr() as *const GLvoid,
      gl::STATIC_DRAW,
    );

    let pos_attr = gl.GetAttribLocation(cube_program, cstr!("Position"));
    gl.EnableVertexAttribArray(pos_attr as u32);
    gl.VertexAttribPointer(
      pos_attr as u32,
      3,
      gl::FLOAT,
      gl::FALSE,
      (4 * std::mem::size_of::<f32>()) as i32,
      std::ptr::null(),
    );

    let texture_coords_attr = gl.GetAttribLocation(cube_program, cstr!("TexCoords"));
    gl.EnableVertexAttribArray(texture_coords_attr as u32);
    gl.VertexAttribPointer(
      texture_coords_attr as u32,
      2,
      gl::FLOAT,
      gl::FALSE,
      (4 * std::mem::size_of::<f32>()) as i32,
      (2 * std::mem::size_of::<f32>()) as *const GLvoid,
    );

    (vao, vbo, ebo)
  };

  let (screen_vao, screen_vbo) = unsafe {
    let (mut vao, mut vbo) = (0, 0);
    gl.GenVertexArrays(1, &mut vao);
    gl.GenBuffers(1, &mut vbo);
    gl.BindVertexArray(vao);
    gl.BindBuffer(gl::ARRAY_BUFFER, vbo);
    gl.BufferData(
      gl::ARRAY_BUFFER,
      (quad_vertices.len() * std::mem::size_of::<f32>()) as GLsizeiptr,
      quad_vertices.as_ptr() as *const GLvoid,
      gl::STATIC_DRAW,
    );

    let pos_attr = gl.GetAttribLocation(screen_program, cstr!("Position"));
    gl.EnableVertexAttribArray(pos_attr as u32);
    gl.VertexAttribPointer(
      pos_attr as u32,
      2,
      gl::FLOAT,
      gl::FALSE,
      (4 * std::mem::size_of::<f32>()) as i32, // offset of each point
      std::ptr::null(),
    );

    let texture_coords_attr = gl.GetAttribLocation(screen_program, cstr!("TexCoords"));
    gl.EnableVertexAttribArray(texture_coords_attr as u32);
    gl.VertexAttribPointer(
      texture_coords_attr as u32,
      2,
      gl::FLOAT,
      gl::FALSE,
      (4 * std::mem::size_of::<f32>()) as i32,           // offset of each point
      (2 * std::mem::size_of::<f32>()) as *const GLvoid, // offset of each point
    );

    (vao, vbo)
  };

  let (text_vao, text_vbo, text_ebo) = unsafe {
    let (mut vao, mut vbo, mut ebo) = (0, 0, 0);
    gl.GenVertexArrays(1, &mut vao);
    gl.GenBuffers(1, &mut vbo);
    gl.BindVertexArray(vao);
    gl.BindBuffer(gl::ARRAY_BUFFER, vbo);
    gl.BufferData(
      gl::ARRAY_BUFFER,
      (4 * std::mem::size_of::<MyTextVertex>() * 1000) as GLsizeiptr,
      std::ptr::null(),
      gl::DYNAMIC_DRAW,
    );

    gl.GenBuffers(1, &mut ebo);
    gl.BindBuffer(gl::ELEMENT_ARRAY_BUFFER, ebo);
    gl.BufferData(
      gl::ELEMENT_ARRAY_BUFFER,
      (6 * std::mem::size_of::<u16>() * 1000) as GLsizeiptr,
      std::ptr::null(),
      gl::DYNAMIC_DRAW,
    );

    let pos_tex_attr = gl.GetAttribLocation(text_program, cstr!("PosTex"));
    gl.EnableVertexAttribArray(pos_tex_attr as u32);
    gl.VertexAttribPointer(
      pos_tex_attr as u32,
      4,
      gl::FLOAT,
      gl::FALSE,
      (std::mem::size_of::<MyTextVertex>()) as i32,
      get_offset!(MyTextVertex, pos_tex) as *const GLvoid,
    );
    let color_attr = gl.GetAttribLocation(text_program, cstr!("Color"));
    gl.EnableVertexAttribArray(color_attr as u32);
    gl.VertexAttribPointer(
      color_attr as u32,
      4,
      gl::FLOAT,
      gl::FALSE,
      (std::mem::size_of::<MyTextVertex>()) as i32,
      get_offset!(MyTextVertex, color_rgba) as *const GLvoid,
    );
    gl.BindBuffer(gl::ARRAY_BUFFER, 0);
    gl.BindVertexArray(0);

    (vao, vbo, ebo)
  };

  let cube_texture = unsafe {
    let cube_texture = load_texture(&gl, "container.jpeg")?;
    gl.UseProgram(cube_program);
    gl.Uniform1i(gl.GetUniformLocation(cube_program, cstr!("uTexture")), 0);
    cube_texture
  };

  let (frame_buffer, texture_color_buffer) = unsafe {
    gl.UseProgram(screen_program);
    gl.Uniform1i(gl.GetUniformLocation(screen_program, cstr!("uTexture")), 0);

    let mut frame_buffer = 0;
    gl.GenFramebuffers(1, &mut frame_buffer);
    gl.BindFramebuffer(gl::FRAMEBUFFER, frame_buffer);

    let mut texture_color_buffer = 0;
    gl.GenTextures(1, &mut texture_color_buffer);
    gl.BindTexture(gl::TEXTURE_2D, texture_color_buffer);
    gl.TexImage2D(
      gl::TEXTURE_2D,
      0,
      gl::RGB as i32,
      320,
      200,
      0,
      gl::RGB,
      gl::UNSIGNED_BYTE,
      std::ptr::null(),
    );
    gl.TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::NEAREST as i32);
    gl.TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::NEAREST as i32);
    gl.FramebufferTexture2D(
      gl::FRAMEBUFFER,
      gl::COLOR_ATTACHMENT0,
      gl::TEXTURE_2D,
      texture_color_buffer,
      0,
    );

    let mut rbo = 0;
    gl.GenRenderbuffers(1, &mut rbo);
    gl.BindRenderbuffer(gl::RENDERBUFFER, rbo);
    gl.RenderbufferStorage(gl::RENDERBUFFER, gl::DEPTH24_STENCIL8, 320, 200);
    gl.FramebufferRenderbuffer(gl::FRAMEBUFFER, gl::DEPTH_STENCIL_ATTACHMENT, gl::RENDERBUFFER, rbo);
    if gl.CheckFramebufferStatus(gl::FRAMEBUFFER) != gl::FRAMEBUFFER_COMPLETE {
      println!("ERROR::FRAMEBUFFER:: Framebuffer is not complete!");
    }
    gl.BindFramebuffer(gl::FRAMEBUFFER, 0);

    (frame_buffer, texture_color_buffer)
  };

  let (atlas_texture, characters) = unsafe {
    let (mut w, mut h) = (0, 0);
    for c in 32..127 {
      if face.load_char(c, ft::face::LoadFlag::RENDER).is_ok() {
        w += face.glyph().bitmap().width();
        h = h.max(face.glyph().bitmap().rows());
      } else {
        eprintln!("could not load character {}", c as u8 as char);
      }
    }

    let mut texture = 0;
    gl.GenTextures(1, &mut texture);
    gl.BindTexture(gl::TEXTURE_2D, texture);
    gl.TexImage2D(
      gl::TEXTURE_2D,
      0,
      gl::RED as i32,
      w,
      h,
      0,
      gl::RED,
      gl::UNSIGNED_BYTE,
      std::ptr::null(),
    );
    gl.TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as i32);
    gl.TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as i32);
    gl.TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::LINEAR as i32);
    gl.TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::LINEAR as i32);

    let mut x = 0;
    let mut characters = std::collections::HashMap::<char, Character>::new();
    gl.PixelStorei(gl::UNPACK_ALIGNMENT, 1);

    for c in 32..127 {
      if face.load_char(c, ft::face::LoadFlag::RENDER).is_ok() {
        gl.TexSubImage2D(
          gl::TEXTURE_2D,
          0,
          x,
          0,
          face.glyph().bitmap().width(),
          face.glyph().bitmap().rows(),
          gl::RED,
          gl::UNSIGNED_BYTE,
          face.glyph().bitmap().buffer().as_ptr() as *const GLvoid,
        );

        println!(
          "{}, {}, {}, {}",
          c as u8 as char,
          x as f32 / w as f32,
          (x as f32 + face.glyph().bitmap().width() as f32) / w as f32,
          face.glyph().bitmap().rows() as f32 / h as f32
        );

        let character = Character {
          tx: x as f32 / w as f32,
          tx_1: (x as f32 + face.glyph().bitmap().width() as f32) / w as f32,
          ty: face.glyph().bitmap().rows() as f32 / h as f32,
          width: face.glyph().bitmap().width() as f32,
          height: face.glyph().bitmap().rows() as f32,
          bearing: glam::vec2(face.glyph().bitmap_left() as f32, face.glyph().bitmap_top() as f32),
          advance: (face.glyph().advance().x >> 6) as f32,
        };
        characters.insert(c as u8 as char, character);

        x += face.glyph().bitmap().width();
      } else {
        eprintln!("could not load character {}", c as u8 as char);
      }
    }

    gl.BindTexture(gl::TEXTURE_2D, 0);

    (texture, characters)
  };

  let build_text = |text: &str, mut x: f32, y: f32, scale: f32, color: glam::Vec3, buf: &mut Vec<MyTextVertex>, ind: &mut Vec<u16>| {
    let mut offset = buf.len() as u16;
    for c in text.chars() {
      let color_rgba = glam::Vec4::from((color, 1.0)).to_array();
      let ch = characters.get(&c).unwrap();
      let x_pos = x + ch.bearing.x as f32 * scale;
      let y_pos = y - (ch.height - ch.bearing.y) * scale;
      let w = ch.width as f32 * scale;
      let h = ch.height as f32 * scale;
      let mut v = (0..4usize)
        .map(|i| {
          MyTextVertex {
            pos_tex: match i {
              0 => [x_pos + w, y_pos + h, ch.tx_1, 0.0], // top right
              1 => [x_pos + w, y_pos, ch.tx_1, ch.ty], // bottom right
              2 => [x_pos, y_pos, ch.tx, ch.ty], // bottom left
              3 => [x_pos, y_pos + h, ch.tx, 0.0], // top left
              _ => panic!("that's too many vertices!"),
            },
            color_rgba,
          }
        })
        .collect::<Vec<_>>();

      let mut indices = vec![
        0u16 + offset, // top right
        1 + offset, // bottom right
        3 + offset, // top left
        //
        1 + offset, // bottom right
        2 + offset, // bottom left
        3 + offset, // top left
      ];

      buf.append(&mut v);
      ind.append(&mut indices);
      x += ch.advance * scale;
      offset += 4;
    }
  };

  let mut event_pump = sdl_context.event_pump().unwrap();
  let timer = sdl_context.timer().unwrap();
  let mut camera_pos = glam::Vec3::new(0.0, 0.0, 3.0);
  let camera_front = glam::Vec3::new(0.0, 0.0, -1.0);
  let camera_up = glam::Vec3::new(0.0, 1.0, 0.0);
  let camera_speed = 2.5;
  let mut camera_zoom = 1.0;
  let mut last = 0.0;
  let (mut viewport_w, mut viewport_h) = (800, 600);
  let mut text_vertices: Vec<MyTextVertex> = Vec::new();
  let mut text_indices: Vec<u16> = Vec::new();

  'running: loop {
    let seconds = timer.ticks() as f32 / 1000.0;
    let delta = seconds - last;
    last = seconds;

    println!("{}", 1.0 / delta);

    for event in event_pump.poll_iter() {
      match event {
        Event::Quit { .. }
        | Event::KeyDown {
          keycode: Some(Keycode::Escape),
          ..
        } => {
          break 'running;
        }
        Event::KeyDown {
          keycode: Some(Keycode::Q),
          ..
        } => camera_zoom += camera_speed * delta,
        Event::KeyDown {
          keycode: Some(Keycode::E),
          ..
        } => camera_zoom -= camera_speed * delta,
        Event::KeyDown {
          keycode: Some(Keycode::W),
          ..
        } => camera_pos += camera_up * camera_speed * delta,
        Event::KeyDown {
          keycode: Some(Keycode::S),
          ..
        } => camera_pos -= camera_up * camera_speed * delta,
        Event::KeyDown {
          keycode: Some(Keycode::A),
          ..
        } => camera_pos -= camera_front.cross(camera_up).normalize() * camera_speed * delta,
        Event::KeyDown {
          keycode: Some(Keycode::D),
          ..
        } => camera_pos += camera_front.cross(camera_up).normalize() * camera_speed * delta,
        Event::Window {
          win_event: WindowEvent::Resized(w, h),
          ..
        } => {
          viewport_w = w;
          viewport_h = h;
        }
        _ => {}
      }
    }

    unsafe {
      gl.BindFramebuffer(gl::FRAMEBUFFER, frame_buffer);
      gl.Viewport(0, 0, 320, 200);
      gl.Enable(gl::DEPTH_TEST);
      gl.ClearColor(0.1, 0.1, 0.1, 1.0);
      gl.Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);

      gl.BindBuffer(gl::ARRAY_BUFFER, cube_vbo);
      gl.BindBuffer(gl::ELEMENT_ARRAY_BUFFER, cube_ebo);
      gl.UseProgram(cube_program);
      gl.BindVertexArray(cube_vao);
      gl.ActiveTexture(gl::TEXTURE0);
      gl.BindTexture(gl::TEXTURE_2D, cube_texture);

      let view = glam::Mat4::look_at_rh(camera_pos, camera_pos + camera_front, camera_up);
      let projection =
        glam::Mat4::orthographic_rh_gl(-300.0 * 0.5, 300.0 * 0.5, -200.0 * 0.5, 200.0 * 0.5, -100.0, 100.0)
          * glam::Mat4::from_scale(glam::Vec3::new(camera_zoom, camera_zoom, 1.0));

      for (i, pos) in CUBE_POSITIONS.iter().enumerate() {
        let mvp_mat = {
          let model = glam::Mat4::from_rotation_translation(
            glam::Quat::from_axis_angle(
              glam::Vec3::new(0.5, 1.0, 0.0).normalize(),
              i as f32 * seconds * 20.0f32.to_radians(),
            ),
            glam::Vec3::from(*pos),
          );
          projection * view * model
        };
        gl.UniformMatrix4fv(
          gl.GetUniformLocation(cube_program, cstr!("uMVP")),
          1,
          gl::FALSE,
          mvp_mat.to_cols_array().as_ptr(),
        );

        gl.DrawElements(
          gl::TRIANGLES,
          CUBE_INDICES.len() as i32,
          gl::UNSIGNED_SHORT,
          std::ptr::null(),
        );
      }

      let mut geometry: VertexBuffers<MyVertex, u16> = VertexBuffers::new();
      for i in -10..10 {
        for j in -10..10 {
          let i = i as f32;
          let j = j as f32;
          let mut tessellator = StrokeTessellator::new();
          let mut options = StrokeOptions::default();
          options.line_width = 0.1;
          let (w, h) = (1.0, 1.0);
          let transform = glam::Mat4::from_rotation_translation(
            glam::Quat::from_axis_angle(glam::Vec3::new(0.0, 0.0, 1.0), seconds * 20.0f32.to_radians()),
            glam::Vec3::new(i, j, -1.0),
          ) * glam::Mat4::from_translation(glam::Vec3::new(w / -2.0, h / -2.0, 0.0));
          tessellator
            .tessellate_rectangle(
              &rect(0.0, 0.0, w, h),
              &options,
              &mut BuffersBuilder::new(
                &mut geometry,
                MyVertexConfig {
                  transform,
                  color_rgba: glam::Vec4::new(0.0, 1.0, 0.0, 1.0),
                },
              ),
            )
            .unwrap();
          let (w, h) = (2.0, 2.0);
          let transform = glam::Mat4::from_rotation_translation(
            glam::Quat::from_axis_angle(glam::Vec3::new(0.0, 0.0, 1.0), seconds * 20.0f32.to_radians()),
            glam::Vec3::new(i, j, -40.0),
          ) * glam::Mat4::from_translation(glam::Vec3::new(w / -2.0, h / -2.0, 0.0));
          tessellator
            .tessellate_rectangle(
              &rect(0.0, 0.0, w, h),
              &options,
              &mut BuffersBuilder::new(
                &mut geometry,
                MyVertexConfig {
                  transform,
                  color_rgba: glam::Vec4::new(1.0, 0.0, 0.0, 1.0),
                },
              ),
            )
            .unwrap();
        }
      }

      gl.UseProgram(lyon_program);
      gl.BindVertexArray(lyon_vao);
      gl.BindBuffer(gl::ARRAY_BUFFER, lyon_vbo);
      gl.BufferSubData(
        gl::ARRAY_BUFFER,
        0,
        (geometry.vertices.len() * std::mem::size_of::<MyVertex>()) as GLsizeiptr,
        geometry.vertices.as_ptr() as *const GLvoid,
      );
      gl.BindBuffer(gl::ELEMENT_ARRAY_BUFFER, lyon_ebo);
      gl.BufferSubData(
        gl::ELEMENT_ARRAY_BUFFER,
        0,
        (geometry.indices.len() * std::mem::size_of::<u16>()) as GLsizeiptr,
        geometry.indices.as_ptr() as *const GLvoid,
      );

      let mvp_mat = {
        let model = glam::Mat4::from_rotation_z(seconds * 20.0f32.to_radians());
        let view = glam::Mat4::look_at_rh(camera_pos, camera_pos + camera_front, camera_up);
        projection * view * model
      };

      gl.UniformMatrix4fv(
        gl.GetUniformLocation(lyon_program, cstr!("uMVP")),
        1,
        gl::FALSE,
        mvp_mat.to_cols_array().as_ptr(),
      );

      gl.DrawElements(
        gl::TRIANGLES,
        geometry.indices.len() as i32,
        gl::UNSIGNED_SHORT,
        std::ptr::null(),
      );
      //

      gl.BindFramebuffer(gl::FRAMEBUFFER, 0);
      gl.Viewport(0, 0, viewport_w, viewport_h);
      gl.Disable(gl::DEPTH_TEST);
      gl.UseProgram(screen_program);
      gl.BindVertexArray(screen_vao);
      gl.ActiveTexture(gl::TEXTURE0);
      gl.BindTexture(gl::TEXTURE_2D, texture_color_buffer);
      gl.DrawArrays(gl::TRIANGLES, 0, 6);

      //
      gl.Enable(gl::BLEND);
      gl.BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);

      gl.UseProgram(text_program);

      let projection = glam::Mat4::orthographic_rh_gl(0.0, viewport_w as f32, 0.0, viewport_h as f32, -10.0, 10.0);
      gl.UniformMatrix4fv(
        gl.GetUniformLocation(text_program, cstr!("uProjection")),
        1,
        gl::FALSE,
        projection.to_cols_array().as_ptr(),
      );

      let s = (32..64u8).map(|c| c as char).collect::<String>();
      build_text(
        &s,
        0.0,
        viewport_h as f32 / 2.0 + 50.0,
        1.0,
        glam::vec3(0.0, 1.0, 0.0),
        &mut text_vertices,
        &mut text_indices
      );
      let s = (64..96u8).map(|c| c as char).collect::<String>();
      build_text(
        &s,
        0.0,
        viewport_h as f32 / 2.0,
        1.0,
        glam::vec3(1.0, 0.0, 0.0),
        &mut text_vertices,
        &mut text_indices
      );
      let s = (96..127u8).map(|c| c as char).collect::<String>();
      build_text(
        &s,
        0.0,
        viewport_h as f32 / 2.0 - 50.0,
        1.0,
        glam::vec3(1.0, 1.0, 0.0),
        &mut text_vertices,
        &mut text_indices
      );

      println!("{:?}", text_indices);

      gl.BindTexture(gl::TEXTURE_2D, atlas_texture);
      gl.BindVertexArray(text_vao);

      gl.BindBuffer(gl::ARRAY_BUFFER, text_vbo);
      gl.BufferSubData(
        gl::ARRAY_BUFFER,
        0,
        (text_vertices.len() * std::mem::size_of::<MyTextVertex>()) as GLsizeiptr,
        text_vertices.as_ptr() as *const GLvoid,
      );
      gl.BindBuffer(gl::ELEMENT_ARRAY_BUFFER, text_ebo);
      gl.BufferData(
        gl::ELEMENT_ARRAY_BUFFER,
        (text_indices.len() * std::mem::size_of::<u16>()) as GLsizeiptr,
        text_indices.as_ptr() as *const GLvoid,
        gl::DYNAMIC_DRAW,
      );

      gl.DrawElements(
        gl::TRIANGLES,
        text_indices.len() as i32,
        gl::UNSIGNED_SHORT,
        std::ptr::null(),
      );

      gl.BindVertexArray(0);
      gl.BindTexture(gl::TEXTURE_2D, 0);
      gl.Disable(gl::BLEND);
      text_vertices.clear();
      text_indices.clear();
    }

    window.gl_swap_window();
  }

  unsafe {
    gl.DeleteVertexArrays(1, &screen_vao);
    gl.DeleteBuffers(1, &screen_vbo);
    gl.DeleteProgram(screen_program);

    gl.DeleteVertexArrays(1, &cube_vao);
    gl.DeleteBuffers(1, &cube_vbo);
    gl.DeleteBuffers(1, &cube_ebo);
    gl.DeleteBuffers(1, &cube_texture);
    gl.DeleteProgram(cube_program);

    gl.DeleteVertexArrays(1, &lyon_vao);
    gl.DeleteBuffers(1, &lyon_vbo);
    gl.DeleteBuffers(1, &lyon_ebo);
    gl.DeleteProgram(lyon_program);

    gl.DeleteVertexArrays(1, &text_vao);
    gl.DeleteBuffers(1, &text_vbo);
    gl.DeleteBuffers(1, &atlas_texture);
    gl.DeleteProgram(text_program);

    gl.DeleteFramebuffers(1, &frame_buffer);
  }

  Ok(())
}
