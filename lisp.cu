#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>
#include "lisp.h"

__managed__ x_any x_nil;
__managed__ x_any x_true;
__managed__ x_any x_dot;
__managed__ x_any x_left;
__managed__ x_any x_right;
__managed__ x_any x_eof;


x_any new_cell(char* new_name) {
  x_any cell;
  char *name;
  cudaMallocManaged(&cell, sizeof(x_cell));
  assert(cell != NULL);
  cudaMallocManaged(&name, strlen(new_name) + 1);
  assert(name != NULL);
  strcpy(name, new_name);
  name(cell) = name;
  return cell;
}

x_any def_token(const char* new_name) {
  x_any cell;
  cell = new_cell((char*)new_name);
  flags(cell) = TOKEN;
  return cell;
}

hash_table_type hash_table;

int hash(char *name)
/* Return a hash value for this name. */
{
  int value = 0;
  while (*name != '\0')
    value = (value * HASH_MULTIPLIER + *name++) % HASH_TABLE_SIZE;
  return value;
}

x_any lookup(char *name, x_any cell)
/* Return the symbol with this name, or NULL if it is not found. */
{
  if (cell == x_nil)
    return NULL;
  else if (strcmp(name(car(cell)), name) == 0)
    return car(cell);
  else
    return lookup(name, cdr(cell));
}

x_any create_symbol(char *new_name)
{
  x_any cell;
  cell = new_cell(new_name);
  flags(cell) = SYMBOL;
  return cell;
}

void print_list(x_any, FILE*);

void print_cell(x_any cell, FILE *outfile)
{
  if (is_atom(cell))
    fprintf(outfile, "%s", name(cell));
  else {
    putc('(', outfile);
    print_list(cell, outfile);
  }
}

void print_list(x_any cell, FILE *outfile)
{
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


__device__ __host__ x_any x_car(x_any cell)
{
  return car(cell);
}

__device__ __host__ x_any x_cdr(x_any cell)
{
  return cdr(cell);
}

x_any x_cons(x_any cell1, x_any cell2)
{
  x_any cell;
  cudaMallocManaged(&cell, sizeof(x_cell));
  assert(cell != NULL);
  flags(cell) = PAIR;
  car(cell) = cell1;
  cdr(cell) = cell2;
  return cell;
}

void enter(x_any cell)
/* Add this symbol to the hash table. */
{
  int hash_val;

  hash_val = hash(name(cell));
  hash_table[hash_val] = x_cons(cell, hash_table[hash_val]);
}

x_any intern(char *name)
/* Return the symbol with this name, creating it if necessary. */
{
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

x_any x_print(x_any cell)
{
  print_cell(cell, stdout);
  putchar('\n');
  return cell;
}

int length(x_any cell)
{
  if (cell == x_nil)
    return 0;
  else if (is_pair(cell))
    return -1;
  else
    return 1 + length(cdr(cell));
}

x_any eval(x_any);

x_any list_eval(x_any cell)
{
  if (cell == x_nil)
    return x_nil;
  else
    return x_cons(eval(car(cell)), list_eval(cdr(cell)));
}

x_any apply(x_any cell, x_any args)
{
  if (is_symbol(cell))
    return x_cons(cell, args);
  if (is_builtin(cell))
    switch (size(cell)) {
    case 0:
      return ((x_fn0)data(cell))();
    case 1:
      return ((x_fn1)data(cell))(car(args));
    case 2:
      return ((x_fn2)data(cell))(car(args), car(cdr(args)));
    case 3:
      return ((x_fn3)data(cell))(car(args), car(cdr(args)), car(cdr(cdr(args))));
    }
  else if (is_user(cell))
    return apply((x_any)data(cell), args);
  else
    assert(0);
  return x_nil;
}

x_any eval(x_any cell)
{
  if (is_atom(cell))
    return cell;
  else if (is_pair(cell) && (is_symbol(car(cell))))
    return x_cons(car(cell), list_eval(cdr(cell)));
  else
    return apply(car(cell), list_eval(cdr(cell)));
}

x_any intern(char*);

x_any def_builtin(char const *name, void *fn, size_t num_args)
{
  x_any cell;

  cell = intern((char*)name);
  flags(cell) = BUILTIN;
  data(cell) = fn;
  size(cell) = num_args;
  return cell;
}

/* This is the lisp tokenizer; it returns a symbol, or one of `(', `)', `.', or EOF */
x_any ratom(FILE *infile)
{
  int c;
  static char buf[MAX_NAME_LEN];
  char *ptr = buf;

  do {
    c = getc(infile);
    if (c == ';')
      do c = getc(infile); while (c != '\n' && c != EOF);
  } while (isspace(c));
  switch (c) {
  case EOF:
    return x_eof;
  case '(':
    return x_left;
  case ')':
    return x_right;
  case '.':
    return x_dot;
  default:
    *ptr++ = c;
    while ((c = getc(infile)) != EOF && !isspace(c) && c != '(' && c != ')')
      *ptr++ = c;
    /* Return the unused character to the input. */
    if (c != EOF)
      ungetc(c, infile);
    *ptr = '\0';
    return intern(buf);
  }
}

x_any read_sexpr(FILE*);
x_any ratom(FILE*);

x_any read_cdr(FILE *infile)
{
  x_any cdr;
  x_any token;

  cdr = read_sexpr(infile);
  token = ratom(infile);

  if (token == x_right)
    return cdr;
  else
    assert(0);
  return x_nil;
}

x_any read_head(FILE*);

x_any read_tail(FILE *infile)
{
  x_any token;
  x_any temp;

  token = ratom(infile);

  if (is_symbol(token))
    return x_cons(token, read_tail(infile));

  if (token == x_left) {
    temp = read_head(infile);
    return x_cons(temp, read_tail(infile));
  }

  if (token == x_dot)
    return read_cdr(infile);

  if (token == x_right)
    return x_nil;

  if (token == x_eof)
    assert(0);
  return x_nil;
}

x_any read_head(FILE *infile)
{
  x_any token;
  x_any temp;

  token = ratom(infile);
  if (is_symbol(token) || is_builtin(token))
    return x_cons(token, read_tail(infile));
  if (token == x_left) {
    temp = read_head(infile);
    return x_cons(temp, read_tail(infile));
  }
  if (token == x_right)
    return x_nil;
  if (token == x_dot)
    assert(0);
  if (token == x_eof)
    assert(0);
  return x_nil;
}

x_any read_sexpr(FILE *infile)
{
  x_any token;

  token = ratom(infile);
  if (is_symbol(token) || is_builtin(token))
    return token;
  if (token == x_left)
    return read_head(infile);
  if (token == x_right)
    assert(0);
  if (token == x_dot)
    assert(0);
  if (token == x_eof)
    return token;
  return x_nil;
}

void init(void)
{
  x_dot = def_token(".");
  x_left = def_token("(");
  x_right = def_token(")");
  x_eof = def_token("EOF");

  x_nil = create_symbol("nil");
  for (int i = 0; i < HASH_TABLE_SIZE; i++)
    hash_table[i] = x_nil;
  enter(x_nil);
  x_true = intern("true");

  def_builtin("car", (void*)x_car, 1);
  def_builtin("cdr", (void*)x_cdr, 1);
  def_builtin("cons", (void*)x_cons, 2);
  def_builtin("print", (void*)x_print, 1);
}

int main(int argc, const char* argv[])
{
  x_any expr;
  x_any value;

  init();
  for (;;) {
    printf(": ");
    expr = read_sexpr(stdin);
    if (expr == x_eof)
      break;
    value = eval(expr);
    print_cell(value, stdout);
    putchar('\n');
  }
}
