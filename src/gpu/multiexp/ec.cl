typedef struct {
  field x;
  field y;
  bool inf;
  uint _; // WARNING: Padding, so that size of struct gets 104 bytes.
} affine;

typedef struct {
  field x;
  field y;
  field z;
} projective;

projective ec_double(projective inp) {
  if(eq(inp.z, ZERO)) return inp;
  field a = mulmod(inp.x, inp.x); // A = X1^2
  field b = mulmod(inp.y, inp.y); // B = Y1^2
  field c = mulmod(b, b); // C = B^2

  // D = 2*((X1+B)2-A-C)
  field d = addmod(inp.x, b);
  d = mulmod(d, d); d = submod(submod(d, a), c); d = addmod(d, d);

  field e = addmod(addmod(a, a), a); // E = 3*A

  field f = mulmod(e, e);

  inp.z = mulmod(inp.y, inp.z); inp.z = addmod(inp.z, inp.z); // Z3 = 2*Y1*Z1
  inp.x = submod(submod(f, d), d); // X3 = F-2*D

  // Y3 = E*(D-X3)-8*C
  c = addmod(c, c); c = addmod(c, c); c = addmod(c, c);
  inp.y = submod(mulmod(submod(d, inp.x), e), c);

  return inp;
}

projective ec_add_mixed(projective a, affine b) {
  if(b.inf) return a;

  if(eq(a.z, ZERO)) {
    a.x = b.x;
    a.y = b.y;
    a.z = ONE;
    return a;
  }

  field z1z1 = mulmod(a.z, a.z);
  field u2 = mulmod(b.x, z1z1);
  field s2 = mulmod(mulmod(b.y, a.z), z1z1);

  if(eq(a.x, u2) && eq(b.y, s2))
    return ec_double(a);
  else {
    field h = submod(u2, a.x); // H = U2-X1
    field hh = mulmod(h, h); // HH = H^2
    field i = addmod(hh, hh); i = addmod(i, i); // I = 4*HH
    field j = mulmod(h, i); // J = H*I
    field r = submod(s2, a.y); r = addmod(r, r); // r = 2*(S2-Y1)
    field v = mulmod(a.x, i);

    projective ret;

     // X3 = r^2 - J - 2*V
    ret.x = submod(submod(mulmod(r, r), j), addmod(v, v));

     // Y3 = r*(V-X3)-2*Y1*J
    j = mulmod(a.y, j); j = addmod(j, j);
    ret.y = submod(mulmod(submod(v, ret.x), r), j);

    // Z3 = (Z1+H)^2-Z1Z1-HH
    ret.z = addmod(a.z, h); ret.z = submod(submod(mulmod(ret.z, ret.z), z1z1), hh);
    return ret;
  }
}

projective ec_add(projective a, projective b) {
  if(eq(a.z, ZERO)) return b;
  if(eq(b.z, ZERO)) return a;

  field z1z1 = mulmod(a.z, a.z); // Z1Z1 = Z1^2
  field z2z2 = mulmod(b.z, b.z); // Z2Z2 = Z2^2
  field u1 = mulmod(a.x, z2z2); // U1 = X1*Z2Z2
  field u2 = mulmod(b.x, z1z1); // U2 = X2*Z1Z1
  field s1 = mulmod(mulmod(a.y, b.z), z2z2); // S1 = Y1*Z2*Z2Z2
  field s2 = mulmod(mulmod(b.y, a.z), z1z1); // S2 = Y2*Z1*Z1Z1

  if(eq(u1, u2) && eq(s1, s2))
    return ec_double(a);
  else {
    field h = submod(u2, u1); // H = U2-U1
    field i = addmod(h, h); i = mulmod(i, i); // I = (2*H)^2
    field j = mulmod(h, i); // J = H*I
    field r = submod(s2, s1); r = addmod(r, r); // r = 2*(S2-S1)
    field v = mulmod(u1, i); // V = U1*I
    a.x = submod(submod(submod(mulmod(r, r), j), v), v); // X3 = r^2 - J - 2*V

    // Y3 = r*(V - X3) - 2*S1*J
    a.y = mulmod(submod(v, a.x), r);
    s1 = mulmod(s1, j); s1 = addmod(s1, s1); // S1 = S1 * J * 2
    a.y = submod(a.y, s1);

    // Z3 = ((Z1+Z2)^2 - Z1Z1 - Z2Z2)*H
    a.z = addmod(a.z, b.z); a.z = mulmod(a.z, a.z);
    a.z = submod(submod(a.z, z1z1), z2z2);
    a.z = mulmod(a.z, h);

    return a;
  }
}

projective ec_mul(affine a, ulong4 b) {
  ulong *ls = (ulong*)&b;
  projective p = {ZERO, ONE, ZERO};
  for(int i = 3; i >= 0; i--) {
    ulong l = ls[i];
    for(uint j = 0; j < 64; j++) {
      p = ec_double(p);
      if(l >> 63)
        p = ec_add_mixed(p, a);
      l <<= 1;
    }
  }
  return p;
}
