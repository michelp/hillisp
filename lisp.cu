#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>
#include <inttypes.h>
#include "lisp.h"

__managed__ x_any x_nil;
__managed__ x_any x_true;
__managed__ x_any x_dot;
__managed__ x_any x_lparen;
__managed__ x_any x_rparen;
__managed__ x_any x_lbrack;
__managed__ x_any x_rbrack;
__managed__ x_any x_eof;

__managed__ hash_table_type hash_table;

char* new_name(const char* name) {
  char *n;
  cudaMallocManaged(&n, strlen(name) + 1);
  assert(n != NULL);
  strcpy(n, name);
  return n;
}

x_any new_cell(const char* name, uint64_t type) {
  x_any cell;
  cudaMallocManaged(&cell, sizeof(x_cell));
  type(cell) = 0;
  assert(cell != NULL);
  set_cdr(cell, NULL);
  set_car(cell, NULL);
  set_type_flag(cell, type);
  if (name == NULL)
    name(cell) = NULL;
  else
    name(cell) = new_name(name);
  return cell;
}

x_any def_token(const char* new_name) {
  x_any cell;
  cell = new_cell(new_name, X_TOKEN);
  return cell;
}

int hash(const char *name) {
  int value = 0;
  while (*name != '\0')
    value = (value * X_HASH_MULTIPLIER + *name++) % X_HASH_TABLE_SIZE;
  return value;
}

x_any lookup(const char *name, x_any cell) {
  if (cell == x_nil)
    return NULL;
  else if (strcmp(name(car(cell)), name) == 0)
    return car(cell);
  else
    return lookup(name, cdr(cell));
}

x_any create_symbol(const char *new_name) {
  x_any cell;
  cell = new_cell(new_name, X_SYMBOL);
  if (isdigit(new_name[0])) {
    set_car(cell, atol(new_name));
    set_type_flag(cell, X_INT);
  }
  return cell;
}

void print_cell(x_any cell, FILE *outfile) {
  if (is_int(cell))
    fprintf(outfile, "%" PRIu64, int_car(cell));
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

x_any x_is(x_any cell1, x_any cell2) {
  if (cell1 == cell2)
    return x_true;
  return x_nil;
}

x_any x_car(x_any cell) {
  return car(cell);
}

x_any x_cdr(x_any cell) {
  return cdr(cell);
}

x_any x_cons(x_any cell1, x_any cell2) {
  x_any cell;
  cell = new_cell(NULL, X_PAIR);
  set_car(cell, cell1);
  set_cdr(cell, cell2);
  return cell;
}

x_any x_add(x_any cell1, x_any cell2) {
  x_any cell;

  if (is_int(cell1) && is_int(cell2)) {
    cell = new_cell(NULL, X_INT);
    set_car(cell, int_car(cell1) + int_car(cell2));
    return cell;
  }
  assert(0);
  return x_nil;
}

x_any x_min(x_any cell1, x_any cell2) {
  x_any cell;

  if (is_int(cell1) && is_int(cell2)) {
    cell = new_cell(NULL, X_INT);
    set_car(cell, int_car(cell1) - int_car(cell2));
    return cell;
  }
  assert(0);
  return x_nil;
}

x_any x_mul(x_any cell1, x_any cell2) {
  x_any cell;

  if (is_int(cell1) && is_int(cell2)) {
    cell = new_cell(NULL, X_INT);
    set_car(cell, int_car(cell1) * int_car(cell2));
    return cell;
  }
  assert(0);
  return x_nil;
}

x_any x_div(x_any cell1, x_any cell2) {
  x_any cell;

  if (is_int(cell1) && is_int(cell2)) {
    cell = new_cell(NULL, X_INT);
    set_car(cell, int_car(cell1) / int_car(cell2));
    return cell;
  }
  assert(0);
  return x_nil;
}

void enter(x_any cell) {
  int hash_val;

  hash_val = hash(name(cell));
  hash_table[hash_val] = x_cons(cell, hash_table[hash_val]);
}

x_any intern(const char *name) {
  x_any cell;

  cell = lookup(name, hash_table[hash(name)]);
  if (cell != NULL)
    return cell;
  else {
    cell = create_symbol(name);
    enter(cell);
    return cell;
  }
}

x_any x_print(x_any cell) {
  print_cell(cell, stdout);
  putchar('\n');
  return cell;
}

int length(x_any cell) {
  if (cell == x_nil)
    return 0;
  else
    return 1 + length(cdr(cell));
}

x_any list_eval(x_any cell) {
  if (cell == x_nil)
    return x_nil;
  else
    return x_cons(x_eval(car(cell)), list_eval(cdr(cell)));
}

x_any x_apply(x_any cell, x_any args) {
  if (is_symbol(cell) && !(is_func(cell)))
    return x_cons(cell, args);
  if (is_pair(cell))
    return x_cons(x_eval(cell), args);
  if (is_builtin(cell)) {
    if (is_fn0(cell))
      return ((x_fn0)cdr(cell))();
    else if (is_fn1(cell))
      return ((x_fn1)cdr(cell))(car(args));
    else if (is_fn2(cell))
      return ((x_fn2)cdr(cell))(car(args), car(cdr(args)));
    else if (is_fn3(cell))
      return ((x_fn3)cdr(cell))(car(args), car(cdr(args)), car(cdr(cdr(args))));
    else
      assert(0);
  }
  else if (is_user(cell))
    return x_apply((x_any)cdr(cell), args);
  else
    assert(0);
  return x_nil;
}

x_any x_quote(x_any cell) {
  return cell;
}

x_any x_cond(x_any clauses) {
  if (clauses == x_nil)
    return x_nil;
  else if (x_eval(car(car(clauses))) != x_nil)
    return x_eval(car(cdr(clauses)));
  else 
    return x_cond(car(cdr(clauses)));
}

x_any x_eval(x_any cell) {
  if (is_atom(cell))
    return cell;
  else if (is_pair(cell) && (is_func(car(cell))))
    return x_apply(car(cell), list_eval(cdr(cell)));
  else
    return x_cons(car(cell), list_eval(cdr(cell)));
}

x_any def_builtin(char const *name, void *fn, size_t num_args) {
  x_any cell;

  cell = intern(name);
  set_type_flag(cell, X_BUILTIN);
  set_cdr(cell, fn);
  switch(num_args) {
  case 0:
    set_type_flag(cell, X_FN0);
    break;
  case 1:
    set_type_flag(cell, X_FN1);
    break;
  case 2:
    set_type_flag(cell, X_FN2);
    break;
  case 3:
    set_type_flag(cell, X_FN3);
    break;
  }
  return cell;
}

x_any read_token(FILE *infile) {
  int c;
  static char buf[X_MAX_NAME_LEN];
  char *ptr = buf;

  do {
    c = getc(infile);
    if (c == '#')
      do c = getc(infile); while (c != '\n' && c != EOF);
  } while (isspace(c));
  switch (c) {
  case EOF:
    return x_eof;
  case '(':
    return x_lparen;
  case ')':
    return x_rparen;
  case '[':
    return x_lbrack;
  case ']':
    return x_rbrack;
  case '.':
    return x_dot;
  default:
    *ptr++ = c;
    while ((c = getc(infile)) != EOF &&
           !isspace(c) &&
           c != '(' && c != ')' &&
           c != '[' && c!= ']')
      *ptr++ = c;
    if (c != EOF)
      ungetc(c, infile);
    *ptr = '\0';
    return intern(buf);
  }
}

x_any read_cdr(FILE *infile) {
  x_any cdr;
  x_any token;

  cdr = read_sexpr(infile);
  token = read_token(infile);

  if (token == x_rparen)
    return cdr;
  else
    assert(0);
  return x_nil;
}

x_any read_tail(FILE *infile) {
  x_any token;
  x_any temp;

  token = read_token(infile);

  if (is_symbol(token) || is_builtin(token))
    return x_cons(token, read_tail(infile));

  if (token == x_lparen) {
    temp = read_head(infile);
    return x_cons(temp, read_tail(infile));
  }

  if (token == x_dot)
    return read_cdr(infile);

  if (token == x_rparen)
    return x_nil;

  if (token == x_eof)
    assert(0);
  return x_nil;
}

x_any read_head(FILE *infile) {
  x_any token;
  x_any temp;

  token = read_token(infile);
  if (is_symbol(token) || is_builtin(token))
    return x_cons(token, read_tail(infile));
  if (token == x_lparen) {
    temp = read_head(infile);
    return x_cons(temp, read_tail(infile));
  }
  if (token == x_rparen)
    return x_nil;
  if (token == x_dot)
    assert(0);
  if (token == x_eof)
    assert(0);
  return x_nil;
}

x_any read_xector(FILE *infile) {
  x_any token;
  x_any cell;

  cell = new_cell("xector", X_XECTOR);
  token = read_token(infile);
  while (token != x_rbrack) {
    token = read_token(infile);
  }
  return x_cons(cell, read_tail(infile));
}

x_any read_sexpr(FILE *infile) {
  x_any token;

  token = read_token(infile);
  if (is_symbol(token) || is_builtin(token))
    return token;
  if (token == x_lparen)
    return read_head(infile);
  if (token == x_lbrack)
    return read_xector(infile);
  if (token == x_rparen)
    assert(0);
  if (token == x_dot)
    assert(0);
  if (token == x_eof)
    return token;
  return x_nil;
}

void init(void) {
  x_dot = def_token(".");
  x_lparen = def_token("(");
  x_rparen = def_token(")");
  x_lbrack = def_token("[");
  x_rbrack = def_token("]");
  x_eof = def_token("EOF");

  x_nil = create_symbol("nil");
  for (int i = 0; i < X_HASH_TABLE_SIZE; i++)
    hash_table[i] = x_nil;
  enter(x_nil);
  x_true = intern("true");

  def_builtin("is", (void*)x_is, 2);
  def_builtin("car", (void*)x_car, 1);
  def_builtin("cdr", (void*)x_cdr, 1);
  def_builtin("cons", (void*)x_cons, 2);
  def_builtin("quote", (void*)x_quote, 1);
  def_builtin("cond", (void*)x_cond, 1);
  def_builtin("eval", (void*)x_eval, 1);
  def_builtin("apply", (void*)x_apply, 2);
  def_builtin("print", (void*)x_print, 1);

  def_builtin("+", (void*)x_add, 2);
  def_builtin("-", (void*)x_min, 2);
  def_builtin("*", (void*)x_mul, 2);
  def_builtin("/", (void*)x_div, 2);
}

int main(int argc, const char* argv[]) {
  x_any expr;
  x_any value;

  init();
  for (;;) {
    printf("? ");
    expr = read_sexpr(stdin);
    if (expr == x_eof)
      break;
    value = x_eval(expr);
    printf(": ");
    print_cell(value, stdout);
    putchar('\n');
  }
}
