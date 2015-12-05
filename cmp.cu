#include "lisp.h"

x_any x_eq(x_any cell1, x_any cell2) {
  x_any cell;
  if (is_int(cell1) && is_int(cell2)) {
    if (int64_car(cell1) == int64_car(cell2))
      return x_true;
  }
  else if (is_xector(cell1) && is_xector(cell2)) {
    assert(xector_size(cell1) == xector_size(cell2));
    cell = new_xector(NULL, xector_size(cell1));
    SYNCS(stream);
    xd_eq_xint64<<<GRIDBLOCKS(xector_size(cell1)), THREADSPERBLOCK, 0, stream>>>
      (int64_cars(cell1), int64_cars(cell2), int64_cars(cell), xector_size(cell1));
    CHECK;
    return cell;
  }
  else if (is_atom(cell1) && is_atom(cell2)) {
    if (strcmp(cell1->name, cell2->name) == 0)
      return x_true;
  }
  else if (is_pair(cell1) && is_pair(cell2)) {
    do {
      if (x_eq(car(cell1), car(cell2)) == x_nil)
        return x_nil;
      cell1 = cdr(cell1);
      cell2 = cdr(cell2);
    } while (is_pair(cell1) && is_pair(cell2));
    if (x_eq(cell1, cell2) != x_nil)
      return x_true;
  }
  return x_nil;
}

x_any x_neq(x_any cell1, x_any cell2) {
  if (x_eq(cell1, cell2) == x_true)
    return x_nil;
  return x_true;
}

x_any x_gt(x_any cell1, x_any cell2) {
  if (is_int(cell1) && is_int(cell2)) {
    if (int64_car(cell1) > int64_car(cell2))
      return x_true;
  }
  else
    if (strcmp(cell1->name, cell2->name) > 0)
      return x_true;
  return x_nil;
}

x_any x_lt(x_any cell1, x_any cell2) {
  if (is_int(cell1) && is_int(cell2)) {
    if (int64_car(cell1) < int64_car(cell2))
      return x_true;
  }
  else
    if (strcmp(cell1->name, cell2->name) < 0)
      return x_true;
  return x_nil;
}
