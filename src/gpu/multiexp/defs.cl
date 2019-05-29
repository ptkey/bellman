// Bls12-381, F_q
// Size: 384-bits (12 limbs)
// Modulus: 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab

#define LIMBS 12
#define P ((field){{0xffffaaab,0xb9feffff,0xb153ffff,0x1eabfffe,0xf6b0f624,0x6730d2a0,0xf38512bf,0x64774b84,0x434bacd7,0x4b1ba7b6,0x397fe69a,0x1a0111ea}})
#define ONE ((field){{0x0002fffd,0x76090000,0xc40c0002,0xebf4000b,0x53c758ba,0x5f489857,0x70525745,0x77ce5853,0xa256ec6d,0x5c071a97,0xfa80e493,0x15f65ec3}})
#define ZERO ((field){{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}})
#define INV ((uint32)4294770685)
