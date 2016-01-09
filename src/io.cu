#include "lisp.h"

char* x_str(x_any cell) {
  char *bp;
  size_t size;
  FILE *stream;
  stream = open_memstream (&bp, &size);
  print_cell(cell, stream);
  fclose(stream);
  return bp;
}

void print_el(FILE* outfile, x_any cell, int i) {
  assert(is_ixector(cell) || is_dxector(cell));
  if (is_ixector(cell))
    fprintf(outfile, "%" PRIi64, xector_car_ith(cell, i));
  else if (is_dxector(cell))
    fprintf(outfile, "%9.16f", xector_car_dth(cell, i));
}

void print_xector(x_any cell, FILE *outfile) {
  int i;
  putc('[', outfile);
  SYNCS(x_env.stream);
  if (xector_size(cell) < 1024) {
    for (int i = 0; i < xector_size(cell); i++) {
      print_el(outfile, cell, i);
      if (i != (xector_size(cell) - 1))
        putc(' ', outfile);
    }
  }
  else {
    for (i = 0; i < 15; i++) {
      print_el(outfile, cell, i);
      if (i != (xector_size(cell) - 1))
        putc(' ', outfile);
    }
    fprintf(outfile, "... ");
    for (i = xector_size(cell) - 15; i < xector_size(cell); i++) {
      print_el(outfile, cell, i);
      if (i != (xector_size(cell) - 1))
        putc(' ', outfile);
    }
  }
  putc(']', outfile);
}

void print_cell(x_any cell, FILE *outfile) {
  if (is_builtin(cell) || is_special(cell))
    fprintf(outfile, "<%s at %p>", sval(type(cell)), (void*)val(cell));
  else if (is_binding(cell))
    fprintf(outfile, "%s", sval(cell));
  else if (is_user(cell)) {
    putc('(', outfile);
    print_list(cell, outfile);
  }
  else if (is_int(cell))
    fprintf(outfile, "%" PRIi64, ival(cell));
  else if (is_double(cell))
    fprintf(outfile, "%9.16f", dval(cell));
  else if (is_dcomplex(cell))
    fprintf(outfile, "(%9.16f+%9.16fi)", crval(cell), cival(cell));
  else if (is_xector(cell))
    print_xector(cell, outfile);
  else if (is_symbol(cell))
    fprintf(outfile, "%s", sval(cell));
  else if (is_pair(cell)) {
    putc('(', outfile);
    print_list(cell, outfile);
  }
  else
    assert(0);
}

void print_list(x_any cell, FILE *outfile) {
  print_cell(car(cell), outfile);
  if (cdr(cell) == x_env.nil)
    putc(')', outfile);
  else if (!is_pair(cdr(cell)) ) {
    fprintf(outfile, " . ");
    print_cell(cdr(cell), outfile);
    putc(')', outfile);
  }
  else {
    putc(' ', outfile);
    print_list(cdr(cell), outfile);
  }
}

x_any x_print(x_any cell) {
  print_cell(cell, stdout);
  return cell;
}

x_any x_println(x_any cell) {
  print_cell(cell, stdout);
  putchar('\n');
  return cell;
}

x_any x_printsp(x_any cell) {
  print_cell(cell, stdout);
  putchar(' ');
  return cell;
}
