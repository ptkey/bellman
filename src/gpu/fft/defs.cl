// Bls12-381, F_r
// Size: 256-bits (8 limbs)
// Modulus: 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001

#define LIMBS 8
#define P ((field){{0x00000001,0xffffffff,0xfffe5bfe,0x53bda402,0x09a1d805,0x3339d808,0x299d7d48,0x73eda753}})
#define ONE ((field){{0xfffffffe,0x00000001,0x00034802,0x5884b7fa,0xecbc4ff5,0x998c4fef,0xacc5056f,0x1824b159}})
#define ZERO ((field){{0, 0, 0, 0, 0, 0, 0, 0}})
#define INV ((uint32)4294967295)
