#include "lisp.h"

x_any x_add(x_any cell1, x_any cell2) {
  x_any cell;
  if (is_int(cell1) && is_int(cell2)) {
    cell = new_cell(NULL, x_int);
    set_car(cell, int64_car(cell1) + int64_car(cell2));
    return cell;
  }
  else if (is_xector(cell1) && is_xector(cell2)) {
    assert(xector_size(cell1) == xector_size(cell2));
    cell = new_xector(NULL, xector_size(cell1));
    SYNCS(stream);
    xd_add_xint64<<<GRIDBLOCKS(xector_size(cell1)), THREADSPERBLOCK, 0, stream>>>
      (int64_cars(cell1), int64_cars(cell2), int64_cars(cell), xector_size(cell1));
    CHECK;
    return cell;
  }
  assert(0);
  return x_nil;
}

x_any x_sub(x_any cell1, x_any cell2) {
  x_any cell;
  if (is_int(cell1) && is_int(cell2)) {
    cell = new_cell(NULL, x_int);
    set_car(cell, int64_car(cell1) - int64_car(cell2));
    return cell;
  }
  else if (is_xector(cell1) && is_xector(cell2)) {
    assert(xector_size(cell1) == xector_size(cell2));
    cell = new_xector(NULL, xector_size(cell1));
    SYNCS(stream);
    xd_sub_xint64<<<GRIDBLOCKS(xector_size(cell1)), THREADSPERBLOCK, 0, stream>>>
      (int64_cars(cell1), int64_cars(cell2), int64_cars(cell), xector_size(cell1));
    CHECK;
    return cell;
  }
  assert(0);
  return x_nil;
}

x_any x_mul(x_any cell1, x_any cell2) {
  x_any cell;
  if (is_int(cell1) && is_int(cell2)) {
    cell = new_cell(NULL, x_int);
    set_car(cell, int64_car(cell1) * int64_car(cell2));
    return cell;
  }
  else if (is_xector(cell1) && is_xector(cell2)) {
    assert(xector_size(cell1) == xector_size(cell2));
    cell = new_xector(NULL, xector_size(cell1));
    SYNCS(stream);
    xd_mul_xint64<<<GRIDBLOCKS(xector_size(cell1)), THREADSPERBLOCK, 0, stream>>>
      (int64_cars(cell1), int64_cars(cell2), int64_cars(cell), xector_size(cell1));
    CHECK;
    return cell;
  }
  assert(0);
  return x_nil;
}

x_any x_div(x_any cell1, x_any cell2) {
  x_any cell;
  if (is_int(cell1) && is_int(cell2)) {
    cell = new_cell(NULL, x_int);
    set_car(cell, int64_car(cell1) / int64_car(cell2));
    return cell;
  }
  else if (is_xector(cell1) && is_xector(cell2)) {
    assert(xector_size(cell1) == xector_size(cell2));
    cell = new_xector(NULL, xector_size(cell1));
    SYNCS(stream);
    xd_div_xint64<<<GRIDBLOCKS(xector_size(cell1)), THREADSPERBLOCK, 0, stream>>>
      (int64_cars(cell1), int64_cars(cell2), int64_cars(cell), xector_size(cell1));
    CHECK;
    return cell;
  }
  assert(0);
  return x_nil;
}

x_any x_fma(x_any cell1, x_any cell2, x_any cell3) {
  x_any cell;
  if (is_int(cell1) && is_int(cell2)) {
    cell = new_cell(NULL, x_int);
    set_car(cell, int64_car(cell1) * int64_car(cell2) + int64_car(cell3));
    return cell;
  }
  else if (is_xector(cell1) && is_xector(cell2) && is_xector(cell3)) {
    SYNCS(stream);
    xd_fma_xint64<<<GRIDBLOCKS(xector_size(cell1)), THREADSPERBLOCK, 0, stream>>>
      (int64_cars(cell1), int64_cars(cell2), int64_cars(cell3), xector_size(cell1));
    CHECK;
    return cell3;
  }
  assert(0);
  return x_nil;
}
