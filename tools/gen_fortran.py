# the fortran interface module from the declarations in cuokada.h: python gen_fortran.py include/cuokada/cuokada.h
import re
out = []
import sys

for line in open(sys.argv[1]):
    m = re.match(r"CUOKADA_API int (\w+)\((.*)\);", line.strip())
    if not m or m.group(2).strip() == "void":
        continue
    name, args = m.group(1), [a.strip() for a in m.group(2).split(",")]
    cuda = name.startswith("cuokada_cuda_")
    names, decls = [], []
    for a in args:
        t, v = a.rsplit(" ", 1)
        stars = len(v) - len(v.lstrip("*"))
        ptr = stars > 0
        v = v.lstrip("*")
        names.append(v)
        real = "real(c_double)" if "double" in t else "real(c_float)"
        if stars == 2:
            decls.append(f"type(c_ptr), intent(out) :: {v}")
        elif t.endswith("void") or (ptr and cuda and "int" not in t):
            decls.append(f"type(c_ptr), value :: {v}")
        elif ptr and t == "int":
            decls.append(f"integer(c_int), intent(out) :: {v}")
        elif ptr:
            intent = "in" if t.startswith("const") else "out"
            decls.append(f"{real}, intent({intent}) :: {v}(*)")
        elif t == "int64_t":
            decls.append(f"integer(c_int64_t), value :: {v}")
        elif t == "int":
            decls.append(f"integer(c_int), value :: {v}")
        else:
            decls.append(f"{real}, value :: {v}")
    head = f"    integer(c_int) function {name}({', '.join(names)}) bind(c, name=\"{name}\")"
    out.append(head if len(head) <= 130 else f"    integer(c_int) function {name}( &\n        {', '.join(names)}) &\n        bind(c, name=\"{name}\")")
    out.append("      import :: c_int, c_int64_t, c_double, c_float, c_ptr")
    out += [f"      {d}" for d in decls]
    out.append(f"    end function {name}\n")
body = "\n".join(out)
print(f"""! -*- f90 -*-
!
! cuokada: elastic dislocations in a half space, on the cpu and the gpu; the fortran interface
! generated from include/cuokada/cuokada.h by tools/gen_fortran.py; do not edit
!
! Copyright (c) 2026 California Institute of Technology, MIT License; see LICENSE
!
! The functions of cuokada.h, with the same arguments and layouts: arrays are row major in c, so
! a fortran array obs(3, nobs) holds the stations, src(7, nsrc) the rectangles, slip(3, nsrc) the
! slips, and out(3, nobs) the sum of the displacements; the matrix out(3, nsrc, F, nobs). The
! cuda functions take device pointers, e.g. c_devloc(a) in cuda fortran.

module cuokada
  use, intrinsic :: iso_c_binding, only: c_int, c_int64_t, c_double, c_float, c_ptr
  implicit none

  interface

{body}
    integer(c_int) function cuokada_has_cuda() bind(c, name="cuokada_has_cuda")
      import :: c_int
    end function cuokada_has_cuda

  end interface

end module cuokada

! end of file""")
