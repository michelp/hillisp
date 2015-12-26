#include "lisp.h"

x_any read_token(FILE *infile) {
  int c;
  static char buf[X_MAX_NAME_LEN];
  char *ptr = buf;
  x_any cell;

  do {
    c = getc(infile);
    if (c == ';')
      do c = getc(infile); while (c != '\n' && c != EOF);
  } while (isspace(c));
  switch (c) {
  case EOF:
    return x_env.eof;
  case '(':
    return x_env.lparen;
  case ')':
    return x_env.rparen;
  case '[':
    return x_env.lbrack;
  case ']':
    return x_env.rbrack;
  case '.':
    return x_env.dot;
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
    if (strcmp(buf, "symbol") == 0)
      return x_env.symbol;

    if (isdigit(buf[0]) || (buf[0] == '-' && isdigit(buf[1]))) {
      cell = new_int(atoll(buf));
      return cell;
    }
    cell = lookup(buf);
    if (cell != NULL)
      return car(cell);
    return new_cell(buf, x_env.symbol);
  }
}

x_any read_cdr(FILE *infile) {
  x_any cdr;
  x_any token;
  cdr = read_sexpr(infile);
  token = read_token(infile);
  if (token == x_env.rparen)
    return cdr;
  else
    assert(0);
  return x_env.nil;
}

x_any read_sexpr_tail(FILE *infile) {
  x_any token;
  x_any temp;
  token = read_token(infile);
  if (is_atom(token))
    return x_cons(token, read_sexpr_tail(infile));
  if (token == x_env.lparen) {
    temp = read_sexpr_head(infile);
    return x_cons(temp, read_sexpr_tail(infile));
  }
  if (token == x_env.lbrack) {
    temp = read_xector(infile);
    return x_cons(temp, read_sexpr_tail(infile));
  }
  if (token == x_env.dot)
    return read_cdr(infile);
  if (token == x_env.rparen)
    return x_env.nil;
  if (token == x_env.eof)
    assert(0);
  return x_env.nil;
}

x_any read_sexpr_head(FILE *infile) {
  x_any token;
  x_any temp;
  token = read_token(infile);
  if (is_atom(token))
    return x_cons(token, read_sexpr_tail(infile));
  else if (token == x_env.lparen) {
    temp = read_sexpr_head(infile);
    return x_cons(temp, read_sexpr_tail(infile));
  }
  else if (token == x_env.lbrack) {
    temp = read_xector(infile);
    return x_cons(temp, read_sexpr_tail(infile));
  }
  else if (token == x_env.rparen)
    return x_env.nil;
  else if (token == x_env.dot)
    assert(0);
  else if (token == x_env.eof)
    assert(0);
  return x_env.nil;
}

x_any read_xector(FILE *infile) {
  x_any val;
  x_any cell;
  x_any typ = NULL;
  size_t size = 0;
  cell = new_xector<int64_t>(NULL, X_XECTOR_BLOCK_SIZE);
  do {
    val = x_eval(read_sexpr(infile));
    if (val == x_env.nil)
      break;
    if (typ == NULL)
      typ = type(val);
    else if (type(val) != typ)
      assert(0); // must all be same type

    if (typ == x_env.int_)
      xector_set_car_ith(cell, size, ival(val));
    else if (typ == x_env.xector)
      xector_set_car_ith(cell, size, car(val));
    else
      assert(0);
    size++;
  } while (1);
  set_car(cell, new_int(size));
  return cell;
}

x_any read_sexpr(FILE *infile) {
  x_any token;
  token = read_token(infile);
  if (is_atom(token))
    return token;
  if (token == x_env.lbrack)
    return read_xector(infile);
  if (token == x_env.lparen)
    return read_sexpr_head(infile);
  if (token == x_env.rparen)
    assert(0);
  if (token == x_env.dot)
    assert(0);
  if (token == x_env.eof)
    return token;
  return x_env.nil;
}

