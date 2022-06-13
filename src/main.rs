#[allow(clippy::all)]
mod gl {
  include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

use gl::types::*;
use image::GenericImageView;
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
const CUBE_VERTICES: [f32; 120] = [
  // positions      //texture coords
  -0.5, -0.5, -0.5, 0.0, 0.0,
  0.5, -0.5, -0.5, 1.0, 0.0,
  0.5, 0.5, -0.5, 1.0, 1.0,
  -0.5, 0.5, -0.5, 0.0, 1.0,
  //
  -0.5, -0.5, 0.5, 0.0, 0.0,
  0.5, -0.5, 0.5, 1.0, 0.0,
  0.5, 0.5, 0.5, 1.0, 1.0,
  -0.5, 0.5, 0.5, 0.0, 1.0,
  //
  -0.5, 0.5, 0.5, 1.0, 0.0,
  -0.5, 0.5, -0.5, 1.0, 1.0,
  -0.5, -0.5, -0.5, 0.0, 1.0,
  -0.5, -0.5, 0.5, 0.0, 0.0,
  //
  0.5, 0.5, 0.5, 1.0, 0.0,
  0.5, 0.5, -0.5, 1.0, 1.0,
  0.5, -0.5, -0.5, 0.0, 1.0,
  0.5, -0.5, 0.5, 0.0, 0.0,
  //
  -0.5, -0.5, -0.5, 0.0, 1.0,
  0.5, -0.5, -0.5, 1.0, 1.0,
  0.5, -0.5, 0.5, 1.0, 0.0,
  -0.5, -0.5, 0.5, 0.0, 0.0,
  //
  -0.5, 0.5, -0.5, 0.0, 1.0,
  0.5, 0.5, -0.5, 1.0, 1.0,
  0.5, 0.5, 0.5, 1.0, 0.0,
  -0.5, 0.5, 0.5, 0.0, 0.0,
];

const CUBE_INDICES: [u32; 36] = [
  0, 1, 3, 1, 2, 3, //
  4, 5, 7, 5, 6, 7, //
  8, 9, 11, 9, 10, 11, //
  12, 13, 15, 13, 14, 15, //
  16, 17, 19, 17, 18, 19, //
  20, 21, 23, 21, 22, 23,
];
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

layout (location = 0) in vec3 Position;
layout (location = 1) in vec2 TexCoords;

uniform mat4 uMVP;

out VERTEX_SHADER_OUTPUT {
  vec2 TexCoords;
} OUT;

void main() {
  gl_Position = uMVP * vec4(Position, 1.0);
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
      gl.GetShaderInfoLog(
        shader,
        len,
        std::ptr::null_mut(),
        error.as_ptr() as *mut GLchar,
      );
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
  let format = if image.color().channel_count() == 3 { gl::RGB } else { gl::RGBA };
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
      gl.GetProgramInfoLog(
        program,
        len,
        std::ptr::null_mut(),
        error.as_ptr() as *mut GLchar,
      );
      return Err(error.to_string_lossy().into_owned());
    }

    gl.DeleteShader(vertex_shader);
    gl.DeleteShader(fragment_shader);

    Ok(program)
  }
}

fn main() -> Result<(), String> {
  let quad_vertices = [
    -1.0f32, 1.0, 0.0, 1.0,
    -1.0, -1.0, 0.0, 0.0,
    1.0, -1.0, 1.0, 0.0,
    -1.0, 1.0, 0.0, 1.0,
    1.0, -1.0, 1.0, 0.0,
    1.0, 1.0, 1.0, 1.0,
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

  let cube_program = {
    let vertex_shader = compile_shader(&gl, VERTEX_SHADER, gl::VERTEX_SHADER)?;
    let fragment_shader = compile_shader(&gl, FRAGMENT_SHADER, gl::FRAGMENT_SHADER)?;
    link_program(&gl, vertex_shader, fragment_shader)?
  };
  let screen_programm = {
    let vertex_shader = compile_shader(&gl, FBO_VERTEX_SHADER, gl::VERTEX_SHADER)?;
    let fragment_shader = compile_shader(&gl, FBO_FRAGMENT_SHADER, gl::FRAGMENT_SHADER)?;
    link_program(&gl, vertex_shader, fragment_shader)?
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
      (CUBE_INDICES.len() * std::mem::size_of::<u32>()) as GLsizeiptr,
      CUBE_INDICES.as_ptr() as *const GLvoid,
      gl::STATIC_DRAW,
    );

    let pos_attr = gl.GetAttribLocation(cube_program, CString::new("Position").unwrap().into_raw());
    gl.EnableVertexAttribArray(pos_attr as u32);
    gl.VertexAttribPointer(
      pos_attr as u32,
      3,
      gl::FLOAT,
      gl::FALSE,
      (5 * std::mem::size_of::<f32>()) as i32, // offset of each point
      std::ptr::null(),
    );

    let texture_coords_attr =
      gl.GetAttribLocation(cube_program, CString::new("TexCoords").unwrap().into_raw());
    gl.EnableVertexAttribArray(texture_coords_attr as u32);
    gl.VertexAttribPointer(
      texture_coords_attr as u32,
      2,
      gl::FLOAT,
      gl::FALSE,
      (5 * std::mem::size_of::<f32>()) as i32, // offset of each point
      (3 * std::mem::size_of::<f32>()) as *const GLvoid, // offset of each point
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

    let pos_attr = gl.GetAttribLocation(screen_programm, CString::new("Position").unwrap().into_raw());
    gl.EnableVertexAttribArray(pos_attr as u32);
    gl.VertexAttribPointer(
      pos_attr as u32,
      2,
      gl::FLOAT,
      gl::FALSE,
      (4 * std::mem::size_of::<f32>()) as i32, // offset of each point
      std::ptr::null(),
    );

    let texture_coords_attr =
      gl.GetAttribLocation(screen_programm, CString::new("TexCoords").unwrap().into_raw());
    gl.EnableVertexAttribArray(texture_coords_attr as u32);
    gl.VertexAttribPointer(
      texture_coords_attr as u32,
      2,
      gl::FLOAT,
      gl::FALSE,
      (4 * std::mem::size_of::<f32>()) as i32, // offset of each point
      (2 * std::mem::size_of::<f32>()) as *const GLvoid, // offset of each point
    );

    (vao, vbo)
  };

  let cube_texture = unsafe {
    let cube_texture = load_texture(&gl, "container.jpeg")?;
    gl.UseProgram(cube_program);
    gl.Uniform1i(
      gl.GetUniformLocation(cube_program, CString::new("uTexture").unwrap().into_raw()),
      0,
    );
    cube_texture
  };

  let (frame_buffer, texture_color_buffer) = unsafe {
    gl.UseProgram(screen_programm);
    gl.Uniform1i(
      gl.GetUniformLocation(screen_programm, CString::new("uTexture").unwrap().into_raw()),
      0,
    );

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
    gl.FramebufferTexture2D(gl::FRAMEBUFFER, gl::COLOR_ATTACHMENT0, gl::TEXTURE_2D, texture_color_buffer, 0);

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

  let mut event_pump = sdl_context.event_pump().unwrap();
  let timer = sdl_context.timer().unwrap();
  let mut camera_pos = glam::Vec3::new(0.0, 0.0, 3.0);
  let camera_front = glam::Vec3::new(0.0, 0.0, -1.0);
  let camera_up = glam::Vec3::new(0.0, 1.0, 0.0);
  let camera_speed = 2.5;
  let mut camera_zoom = 1.0;
  let mut last = 0.0;
  let (mut viewport_w, mut viewport_h) = (800, 600);

  'running: loop {
    let seconds = timer.ticks() as f32 / 1000.0;
    let delta = seconds - last;
    last = seconds;

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
        } =>  {
          viewport_w = w;
          viewport_h = h;
        },
        _ => {}
      }
    }

    unsafe {
      gl.BindFramebuffer(gl::FRAMEBUFFER, frame_buffer);
      gl.Viewport(0, 0, 320, 200);
      gl.Enable(gl::DEPTH_TEST);
      gl.ClearColor(0.1, 0.1, 0.1, 1.0);
      gl.Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);

      gl.UseProgram(cube_program);
      gl.BindVertexArray(cube_vao);
      gl.ActiveTexture(gl::TEXTURE0);
      gl.BindTexture(gl::TEXTURE_2D, cube_texture);

      for (i, pos) in CUBE_POSITIONS.iter().enumerate() {
        let mvp_mat = {
          let model = glam::Mat4::from_rotation_translation(
            glam::Quat::from_axis_angle(
              glam::Vec3::new(0.5, 1.0, 0.0).normalize(),
              i as f32 * seconds * 20.0f32.to_radians(),
            ),
            glam::Vec3::from(*pos),
          );
          let view =
            glam::Mat4::look_at_rh(camera_pos, camera_pos + camera_front, camera_up);
          let aspect = 300.0 / 200.0;
          let projection = glam::Mat4::orthographic_rh_gl(
            -aspect * camera_zoom,
            aspect * camera_zoom,
            -camera_zoom,
            camera_zoom,
            0.1,
            100.0,
          );
          projection * view * model
        };
        gl.UniformMatrix4fv(
          gl.GetUniformLocation(cube_program, CString::new("uMVP").unwrap().into_raw()),
          1,
          gl::FALSE,
          mvp_mat.to_cols_array().as_ptr(),
        );

        gl.DrawElements(
          gl::TRIANGLES,
          CUBE_INDICES.len() as i32,
          gl::UNSIGNED_INT,
          std::ptr::null(),
        );
      }

      gl.BindFramebuffer(gl::FRAMEBUFFER, 0);
      gl.Viewport(0, 0, viewport_w, viewport_h);
      gl.Disable(gl::DEPTH_TEST);
      gl.UseProgram(screen_programm);
      gl.BindVertexArray(screen_vao);
      gl.ActiveTexture(gl::TEXTURE0);
      gl.BindTexture(gl::TEXTURE_2D, texture_color_buffer);
      gl.DrawArrays(gl::TRIANGLES, 0, 6);
    }

    window.gl_swap_window();

    std::thread::sleep(std::time::Duration::new(0, 1_000_000_000u32 / 60));
  }

  unsafe {
    gl.DeleteVertexArrays(1, &cube_vao);
    gl.DeleteVertexArrays(1, &screen_vao);
    gl.DeleteBuffers(1, &cube_vbo);
    gl.DeleteBuffers(1, &cube_ebo);
    gl.DeleteBuffers(1, &cube_texture);
    gl.DeleteBuffers(1, &screen_vbo);
    gl.DeleteProgram(cube_program);
    gl.DeleteProgram(screen_programm);
    gl.DeleteFramebuffers(1, &frame_buffer);
  }

  Ok(())
}
