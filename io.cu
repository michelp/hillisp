#include "lisp.h"

void print_xector(x_any cell, FILE *outfile) {
  int i;
  putc('[', outfile);
  SYNCS(stream);
  if (xector_size(cell) < 1024) {
    for (int i = 0; i < xector_size(cell); i++) {
      fprintf(outfile, "%" PRIi64, xector_car_ith(cell, i));
      if (i != (xector_size(cell) - 1))
        putc(' ', outfile);
    }
  }
  else {
    for (i = 0; i < 15; i++) {
      fprintf(outfile, "%" PRIi64, xector_car_ith(cell, i));
      if (i != (xector_size(cell) - 1))
        putc(' ', outfile);
    }
    fprintf(outfile, " ... ");
    for (i = xector_size(cell) - 15; i < xector_size(cell); i++) {
      fprintf(outfile, "%" PRIi64, xector_car_ith(cell, i));
      if (i != (xector_size(cell) - 1))
        putc(' ', outfile);
    }
  }
  putc(']', outfile);
}

void print_cell(x_any cell, FILE *outfile) {
  if (is_int(cell))
    fprintf(outfile, "%" PRIi64, carr<int64_t>(cell));
  else if (is_xector(cell)) {
    print_xector(cell, outfile);
  }
  else if (is_atom(cell))
    fprintf(outfile, "%s", name(cell));
  else {
    putc('(', outfile);
    print_list(cell, outfile);
  }
}

void print_list(x_any cell, FILE *outfile) {
  print_cell(carr<x_any>(cell), outfile);
  if (cdrr<x_any>(cell) == x_nil)
    putc(')', outfile);
  else if (!is_pair(cdrr<x_any>(cell)) ) {
    fprintf(outfile, " . ");
    print_cell(cdrr<x_any>(cell), outfile);
    putc(')', outfile);
  }
  else {
    putc(' ', outfile);
    print_list(cdrr<x_any>(cell), outfile);
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
