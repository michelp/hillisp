#include "lisp.h"

void print_xector(x_any cell, FILE *outfile) {
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
      for (int i = 0; i < 30; i++) {
        fprintf(outfile, "%" PRIi64, xector_car_ith(cell, i));
        if (i != (xector_size(cell) - 1))
          putc(' ', outfile);
      }
      fprintf(outfile, "... (%zu more)", xector_size(cell) - 30);
    }
    putc(']', outfile);
}

void print_cell(x_any cell, FILE *outfile) {
  if (is_int(cell))
    fprintf(outfile, "%" PRIi64, int64_car(cell));
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
  print_cell(car(cell), outfile);
  if (cdr(cell) == x_nil)
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
