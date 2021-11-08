#[allow(clippy::all)]
mod gl {
  include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

use gl::types::*;
use sdl2::event::{Event, WindowEvent};
use sdl2::keyboard::Keycode;
use sdl2::video::GLProfile;
use std::ffi::CString;
use std::ops::Deref;

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

#[rustfmt::skip]
const TRIANGLE_VERTICIES: [f32; 12] = [
  0.5, 0.5, 0.0,  // top right
  0.5, -0.5, 0.0,  // bottom right
  -0.5, -0.5, 0.0,  // top left
  -0.5, 0.5, 0.0,  // bottom left
];

const TRIANGLE_INDICIES: [u32; 6] = [
  0, 1, 3,
  1, 2, 3
];

const VERTEX_SHADER: &str = r#"
#version 330 core

layout (location = 0) in vec3 Position;

void main() {
  gl_Position = vec4(Position, 1.0);
}
"#;
const FRAGMENT_SHADER: &str = r#"
#version 330 core

out vec4 Color;

uniform vec3 uColor;

void main() {
  Color = vec4(uColor, 1.0f);
}
"#;

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

fn link_program(
  gl: &gl::Gl,
  vertex_shader: GLuint,
  fragment_shader: GLuint,
) -> Result<GLuint, String> {
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

    gl.DetachShader(program, vertex_shader);
    gl.DetachShader(program, fragment_shader);

    Ok(program)
  }
}

fn main() -> Result<(), String> {
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

  let vertex_shader = compile_shader(&gl, VERTEX_SHADER, gl::VERTEX_SHADER)?;
  let fragment_shader = compile_shader(&gl, FRAGMENT_SHADER, gl::FRAGMENT_SHADER)?;
  let program = link_program(&gl, vertex_shader, fragment_shader)?;

  let mut vbo = 0; // Vertex Buffer Object
  let mut vao = 0; // Vertex Array Object
  let mut ebo = 0; // Element Array Object

  unsafe {
    gl.GenBuffers(1, &mut vbo);
    gl.GenBuffers(1, &mut ebo);
    gl.GenVertexArrays(1, &mut vao);

    gl.BindVertexArray(vao);

    gl.BindBuffer(gl::ARRAY_BUFFER, vbo);
    gl.BufferData(
      gl::ARRAY_BUFFER,
      (TRIANGLE_VERTICIES.len() * std::mem::size_of::<f32>()) as GLsizeiptr,
      TRIANGLE_VERTICIES.as_ptr() as *const GLvoid,
      gl::STATIC_DRAW,
    );

    gl.BindBuffer(gl::ELEMENT_ARRAY_BUFFER, ebo);
    gl.BufferData(
      gl::ELEMENT_ARRAY_BUFFER,
      (TRIANGLE_INDICIES.len() * std::mem::size_of::<u32>()) as GLsizeiptr,
      TRIANGLE_INDICIES.as_ptr() as *const GLvoid,
      gl::STATIC_DRAW,
    );

    let pos_attr = gl.GetAttribLocation(program, CString::new("Position").unwrap().into_raw());
    gl.EnableVertexAttribArray(pos_attr as GLuint);
    gl.VertexAttribPointer(
      pos_attr as GLuint,
      3,
      gl::FLOAT,
      gl::FALSE,
      (3 * std::mem::size_of::<f32>()) as GLint, // offset of each point
      std::ptr::null(),
    );
    gl.BindBuffer(gl::ARRAY_BUFFER, 0);
    gl.BindVertexArray(0);
  }

  unsafe {
    gl.Viewport(0, 0, 800, 600);
    gl.ClearColor(0.2, 0.3, 0.3, 1.0);
  }

  let mut event_pump = sdl_context.event_pump().unwrap();
  let timer = sdl_context.timer().unwrap();

  'running: loop {
    for event in event_pump.poll_iter() {
      match event {
        Event::Quit { .. }
        | Event::KeyDown {
          keycode: Some(Keycode::Escape),
          ..
        } => {
          break 'running;
        }
        Event::Window { win_event: WindowEvent::Resized(w, h), .. } => unsafe {
          gl.Viewport(0, 0, w, h);
        },
        _ => {}
      }
    }

    unsafe {
      gl.Clear(gl::COLOR_BUFFER_BIT);
      gl.UseProgram(program);
      let seconds = timer.ticks() as f32 / 1000.0;
      let green_color = f32::sin(seconds) / 2.0 + 0.5;
      let vertex_color_location = gl.GetUniformLocation(program, CString::new("uColor").unwrap().into_raw());
      gl.Uniform3f(vertex_color_location, 0.0, green_color, 0.0);
      gl.BindVertexArray(vao);
      gl.DrawElements(gl::TRIANGLES, 6, gl::UNSIGNED_INT, std::ptr::null());
    }

    window.gl_swap_window();

    ::std::thread::sleep(::std::time::Duration::new(0, 1_000_000_000u32 / 60));
  }

  unsafe {
    gl.DeleteVertexArrays(1, &vao);
    gl.DeleteBuffers(1, &vbo);
    gl.DeleteBuffers(1, &ebo);
    gl.DeleteProgram(program);
    gl.DeleteShader(vertex_shader);
    gl.DeleteShader(fragment_shader);
  }

  Ok(())
}
