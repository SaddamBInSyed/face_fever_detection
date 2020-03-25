from distutils.core import setup, Extension

def main():
    setup(name="pyipccap",
          version="1.0.0",
          description="Python interface for the fputs C library function",
          author="<your name>",
          author_email="your_email@gmail.com",
          ext_modules=[Extension("pyipccap", ["pyipccap.cpp", "op.cpp"],
                       include_dirs=['.'],
                       extra_objects=['libop.a'],
                       extra_link_args = ["-lrt"],
                       extra_compile_args = ["-std=c++14"])]
          )

if __name__ == "__main__":
    main()

