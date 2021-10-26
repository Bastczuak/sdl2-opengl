mod gl {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

use gl::types::*;
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::video::GLProfile;
use std::ffi::CString;
use std::ops::Deref;
use std::ptr;

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
const TRIANGLE_VERTICIES: [GLfloat; 18] = [
  // positions xyz  // colors rgb
  0.5, -0.5, 0.0, 1.0, 0.0, 0.0,
  -0.5, -0.5, 0.0, 0.0, 1.0, 0.0,
  0.0, 0.5, 0.0, 0.0, 0.0, 1.0
];

const VERTEX_SHADER: &'static str = r#"
#version 330 core

layout (location = 0) in vec3 Position;
layout (location = 1) in vec3 Color;

out VERTEX_SHADER_OUTPUT {
  vec3 Color;
} OUT;

void main() {
  gl_Position = vec4(Position, 1.0);
  OUT.Color = Color;
}
"#;
const FRAGMENT_SHADER: &'static str = r#"
#version 330 core

in VERTEX_SHADER_OUTPUT {
  vec3 Color;
} IN;

out vec4 Color;

void main() {
  Color = vec4(IN.Color, 1.0);
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
        gl.ShaderSource(shader, 1, &c_str_src.as_ptr(), ptr::null());
        gl.CompileShader(shader);
        let mut success = gl::TRUE as GLint;
        gl.GetShaderiv(shader, gl::COMPILE_STATUS, &mut success);

        if success == (gl::FALSE as GLint) {
            let mut len = 0;
            gl.GetShaderiv(shader, gl::INFO_LOG_LENGTH, &mut len);
            let error = create_error_buffer(len as usize);
            gl.GetShaderInfoLog(shader, len, ptr::null_mut(), error.as_ptr() as *mut GLchar);
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
            gl.GetProgramInfoLog(program, len, ptr::null_mut(), error.as_ptr() as *mut GLchar);
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

    unsafe {
        gl.GenBuffers(1, &mut vbo);
        gl.BindBuffer(gl::ARRAY_BUFFER, vbo);
        gl.BufferData(
            gl::ARRAY_BUFFER,
            (TRIANGLE_VERTICIES.len() * std::mem::size_of::<GLfloat>()) as GLsizeiptr,
            TRIANGLE_VERTICIES.as_ptr() as *const GLvoid,
            gl::STATIC_DRAW,
        );
        gl.BindBuffer(gl::ARRAY_BUFFER, 0);

        gl.GenVertexArrays(1, &mut vao);
        gl.BindVertexArray(vao);
        gl.BindBuffer(gl::ARRAY_BUFFER, vbo);

        gl.UseProgram(program);
        let pos_attr = gl.GetAttribLocation(program, CString::new("Position").unwrap().into_raw());
        gl.EnableVertexAttribArray(pos_attr as GLuint);
        gl.VertexAttribPointer(
            pos_attr as GLuint,
            3,
            gl::FLOAT,
            gl::FALSE,
            (6 * std::mem::size_of::<GLfloat>()) as GLint, // offset of each point
            std::ptr::null(),
        );
        let color_attr = gl.GetAttribLocation(program, CString::new("Color").unwrap().into_raw());
        gl.EnableVertexAttribArray(color_attr as GLuint);
        gl.VertexAttribPointer(
            color_attr as GLuint,
            3,
            gl::FLOAT,
            gl::FALSE,
            (6 * std::mem::size_of::<GLfloat>()) as GLint, // offset of each point
            (3 * std::mem::size_of::<GLfloat>()) as *const GLvoid, // offset of each color
        );
        gl.BindBuffer(gl::ARRAY_BUFFER, 0);
        gl.BindVertexArray(0);
    }

    unsafe {
        gl.Viewport(0, 0, 800, 600);
        gl.ClearColor(0.7, 0.0, 0.8, 1.0);
    }

    let mut event_pump = sdl_context.event_pump().unwrap();

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
                _ => {}
            }
        }

        unsafe {
            gl.Clear(gl::COLOR_BUFFER_BIT);
            // Draw a triangle from the 3 vertices
            gl.BindVertexArray(vao);
            gl.DrawArrays(gl::TRIANGLES, 0, 3);
        }

        window.gl_swap_window();

        ::std::thread::sleep(::std::time::Duration::new(0, 1_000_000_000u32 / 60));
    }

    unsafe {
        gl.DeleteProgram(program);
        gl.DeleteShader(vertex_shader);
        gl.DeleteShader(fragment_shader);
    }

    Ok(())
}
