#ifndef __SKIP_INTERNAL_FATBINARY_HEADERS
#include "fatbinary_section.h"
#endif
#define __CUDAFATBINSECTION  ".nvFatBinSegment"
#define __CUDAFATBINDATASECTION  ".nv_fatbin"
asm(
".section .nv_fatbin, \"a\"\n"
".align 8\n"
"fatbinData:\n"
".quad 0x00100001ba55ed50,0x00000000000014e8,0x0000004001010002,0x0000000000000310\n"
".quad 0x0000000000000000,0x0000003400010007,0x0000000000000000,0x0000000000000011\n"
".quad 0x0000000000000000,0x0000000000000000,0x33010102464c457f,0x0000000000000007\n"
".quad 0x0000007300be0002,0x0000000000000000,0x0000000000000000,0x00000000000001d0\n"
".quad 0x0000004000340534,0x0001000500400000,0x7472747368732e00,0x747274732e006261\n"
".quad 0x746d79732e006261,0x746d79732e006261,0x78646e68735f6261,0x666e692e766e2e00\n"
".quad 0x65722e766e2e006f,0x6e6f697463612e6c,0x72747368732e0000,0x7274732e00626174\n"
".quad 0x6d79732e00626174,0x6d79732e00626174,0x646e68735f626174,0x6e692e766e2e0078\n"
".quad 0x722e766e2e006f66,0x6f697463612e6c65,0x000000000000006e,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0004000300000032,0x0000000000000000\n"
".quad 0x0000000000000000,0x000000000000004b,0x222f0a1008020200,0x0000000008000000\n"
".quad 0x0000000008080000,0x0000000008100000,0x0000000008180000,0x0000000008200000\n"
".quad 0x0000000008280000,0x0000000008300000,0x0000000008380000,0x0000000008000001\n"
".quad 0x0000000008080001,0x0000000008100001,0x0000000008180001,0x0000000008200001\n"
".quad 0x0000000008280001,0x0000000008300001,0x0000000008380001,0x0000000008000002\n"
".quad 0x0000000008080002,0x0000000008100002,0x0000000008180002,0x0000000008200002\n"
".quad 0x0000000008280002,0x0000000008300002,0x0000000008380002,0x0000002c14000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000300000001,0x0000000000000000,0x0000000000000000,0x0000000000000040\n"
".quad 0x0000000000000041,0x0000000000000000,0x0000000000000001,0x0000000000000000\n"
".quad 0x000000030000000b,0x0000000000000000,0x0000000000000000,0x0000000000000081\n"
".quad 0x0000000000000041,0x0000000000000000,0x0000000000000001,0x0000000000000000\n"
".quad 0x0000000200000013,0x0000000000000000,0x0000000000000000,0x00000000000000c8\n"
".quad 0x0000000000000030,0x0000000200000002,0x0000000000000008,0x0000000000000018\n"
".quad 0x7000000b00000032,0x0000000000000000,0x0000000000000000,0x00000000000000f8\n"
".quad 0x00000000000000d8,0x0000000000000000,0x0000000000000008,0x0000000000000008\n"
".quad 0x0000004001010002,0x0000000000000310,0x0000000000000000,0x0000003c00010007\n"
".quad 0x0000000000000000,0x0000000000000011,0x0000000000000000,0x0000000000000000\n"
".quad 0x33010102464c457f,0x0000000000000007,0x0000007300be0002,0x0000000000000000\n"
".quad 0x0000000000000000,0x00000000000001d0,0x00000040003c053c,0x0001000500400000\n"
".quad 0x7472747368732e00,0x747274732e006261,0x746d79732e006261,0x746d79732e006261\n"
".quad 0x78646e68735f6261,0x666e692e766e2e00,0x65722e766e2e006f,0x6e6f697463612e6c\n"
".quad 0x72747368732e0000,0x7274732e00626174,0x6d79732e00626174,0x6d79732e00626174\n"
".quad 0x646e68735f626174,0x6e692e766e2e0078,0x722e766e2e006f66,0x6f697463612e6c65\n"
".quad 0x000000000000006e,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0004000300000032,0x0000000000000000,0x0000000000000000,0x000000000000004b\n"
".quad 0x222f0a1008020200,0x0000000008000000,0x0000000008080000,0x0000000008100000\n"
".quad 0x0000000008180000,0x0000000008200000,0x0000000008280000,0x0000000008300000\n"
".quad 0x0000000008380000,0x0000000008000001,0x0000000008080001,0x0000000008100001\n"
".quad 0x0000000008180001,0x0000000008200001,0x0000000008280001,0x0000000008300001\n"
".quad 0x0000000008380001,0x0000000008000002,0x0000000008080002,0x0000000008100002\n"
".quad 0x0000000008180002,0x0000000008200002,0x0000000008280002,0x0000000008300002\n"
".quad 0x0000000008380002,0x0000002c14000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000300000001,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000040,0x0000000000000041,0x0000000000000000\n"
".quad 0x0000000000000001,0x0000000000000000,0x000000030000000b,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000081,0x0000000000000041,0x0000000000000000\n"
".quad 0x0000000000000001,0x0000000000000000,0x0000000200000013,0x0000000000000000\n"
".quad 0x0000000000000000,0x00000000000000c8,0x0000000000000030,0x0000000200000002\n"
".quad 0x0000000000000008,0x0000000000000018,0x7000000b00000032,0x0000000000000000\n"
".quad 0x0000000000000000,0x00000000000000f8,0x00000000000000d8,0x0000000000000000\n"
".quad 0x0000000000000008,0x0000000000000008,0x0000004001010002,0x0000000000000310\n"
".quad 0x0000000000000000,0x0000003d00010007,0x0000000000000000,0x0000000000000011\n"
".quad 0x0000000000000000,0x0000000000000000,0x33010102464c457f,0x0000000000000007\n"
".quad 0x0000007300be0002,0x0000000000000000,0x0000000000000000,0x00000000000001d0\n"
".quad 0x00000040003d053d,0x0001000500400000,0x7472747368732e00,0x747274732e006261\n"
".quad 0x746d79732e006261,0x746d79732e006261,0x78646e68735f6261,0x666e692e766e2e00\n"
".quad 0x65722e766e2e006f,0x6e6f697463612e6c,0x72747368732e0000,0x7274732e00626174\n"
".quad 0x6d79732e00626174,0x6d79732e00626174,0x646e68735f626174,0x6e692e766e2e0078\n"
".quad 0x722e766e2e006f66,0x6f697463612e6c65,0x000000000000006e,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0004000300000032,0x0000000000000000\n"
".quad 0x0000000000000000,0x000000000000004b,0x222f0a1008020200,0x0000000008000000\n"
".quad 0x0000000008080000,0x0000000008100000,0x0000000008180000,0x0000000008200000\n"
".quad 0x0000000008280000,0x0000000008300000,0x0000000008380000,0x0000000008000001\n"
".quad 0x0000000008080001,0x0000000008100001,0x0000000008180001,0x0000000008200001\n"
".quad 0x0000000008280001,0x0000000008300001,0x0000000008380001,0x0000000008000002\n"
".quad 0x0000000008080002,0x0000000008100002,0x0000000008180002,0x0000000008200002\n"
".quad 0x0000000008280002,0x0000000008300002,0x0000000008380002,0x0000002c14000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000300000001,0x0000000000000000,0x0000000000000000,0x0000000000000040\n"
".quad 0x0000000000000041,0x0000000000000000,0x0000000000000001,0x0000000000000000\n"
".quad 0x000000030000000b,0x0000000000000000,0x0000000000000000,0x0000000000000081\n"
".quad 0x0000000000000041,0x0000000000000000,0x0000000000000001,0x0000000000000000\n"
".quad 0x0000000200000013,0x0000000000000000,0x0000000000000000,0x00000000000000c8\n"
".quad 0x0000000000000030,0x0000000200000002,0x0000000000000008,0x0000000000000018\n"
".quad 0x7000000b00000032,0x0000000000000000,0x0000000000000000,0x00000000000000f8\n"
".quad 0x00000000000000d8,0x0000000000000000,0x0000000000000008,0x0000000000000008\n"
".quad 0x0000004001010002,0x0000000000000368,0x0000000000000000,0x0000004600010007\n"
".quad 0x0000000000000000,0x0000000000000011,0x0000000000000000,0x0000000000000000\n"
".quad 0x33010102464c457f,0x0000000000000007,0x0000007300be0002,0x0000000000000000\n"
".quad 0x0000000000000000,0x00000000000001e8,0x0000004000460546,0x0001000600400000\n"
".quad 0x7472747368732e00,0x747274732e006261,0x746d79732e006261,0x746d79732e006261\n"
".quad 0x78646e68735f6261,0x666e692e766e2e00,0x67756265642e006f,0x2e00656d6172665f\n"
".quad 0x612e6c65722e766e,0x2e00006e6f697463,0x6261747274736873,0x6261747274732e00\n"
".quad 0x6261746d79732e00,0x6261746d79732e00,0x2e0078646e68735f,0x006f666e692e766e\n"
".quad 0x665f67756265642e,0x766e2e00656d6172,0x7463612e6c65722e,0x00000000006e6f69\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x000500030000003f\n"
".quad 0x0000000000000000,0x0000000000000000,0x000000000000004b,0x222f0a1008020200\n"
".quad 0x0000000008000000,0x0000000008080000,0x0000000008100000,0x0000000008180000\n"
".quad 0x0000000008200000,0x0000000008280000,0x0000000008300000,0x0000000008380000\n"
".quad 0x0000000008000001,0x0000000008080001,0x0000000008100001,0x0000000008180001\n"
".quad 0x0000000008200001,0x0000000008280001,0x0000000008300001,0x0000000008380001\n"
".quad 0x0000000008000002,0x0000000008080002,0x0000000008100002,0x0000000008180002\n"
".quad 0x0000000008200002,0x0000000008280002,0x0000000008300002,0x0000000008380002\n"
".quad 0x0000002c14000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000300000001,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000040,0x000000000000004e,0x0000000000000000,0x0000000000000001\n"
".quad 0x0000000000000000,0x000000030000000b,0x0000000000000000,0x0000000000000000\n"
".quad 0x000000000000008e,0x000000000000004e,0x0000000000000000,0x0000000000000001\n"
".quad 0x0000000000000000,0x0000000200000013,0x0000000000000000,0x0000000000000000\n"
".quad 0x00000000000000e0,0x0000000000000030,0x0000000200000002,0x0000000000000008\n"
".quad 0x0000000000000018,0x0000000100000032,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000110,0x0000000000000000,0x0000000000000000,0x0000000000000001\n"
".quad 0x0000000000000000,0x7000000b0000003f,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000110,0x00000000000000d8,0x0000000000000000,0x0000000000000008\n"
".quad 0x0000000000000008,0x0000004001010002,0x0000000000000368,0x0000000000000000\n"
".quad 0x0000004b00010007,0x0000000000000000,0x0000000000000011,0x0000000000000000\n"
".quad 0x0000000000000000,0x33010102464c457f,0x0000000000000007,0x0000007300be0002\n"
".quad 0x0000000000000000,0x0000000000000000,0x00000000000001e8,0x00000040004b054b\n"
".quad 0x0001000600400000,0x7472747368732e00,0x747274732e006261,0x746d79732e006261\n"
".quad 0x746d79732e006261,0x78646e68735f6261,0x666e692e766e2e00,0x67756265642e006f\n"
".quad 0x2e00656d6172665f,0x612e6c65722e766e,0x2e00006e6f697463,0x6261747274736873\n"
".quad 0x6261747274732e00,0x6261746d79732e00,0x6261746d79732e00,0x2e0078646e68735f\n"
".quad 0x006f666e692e766e,0x665f67756265642e,0x766e2e00656d6172,0x7463612e6c65722e\n"
".quad 0x00000000006e6f69,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x000500030000003f,0x0000000000000000,0x0000000000000000,0x000000000000004b\n"
".quad 0x222f0a1008020200,0x0000000008000000,0x0000000008080000,0x0000000008100000\n"
".quad 0x0000000008180000,0x0000000008200000,0x0000000008280000,0x0000000008300000\n"
".quad 0x0000000008380000,0x0000000008000001,0x0000000008080001,0x0000000008100001\n"
".quad 0x0000000008180001,0x0000000008200001,0x0000000008280001,0x0000000008300001\n"
".quad 0x0000000008380001,0x0000000008000002,0x0000000008080002,0x0000000008100002\n"
".quad 0x0000000008180002,0x0000000008200002,0x0000000008280002,0x0000000008300002\n"
".quad 0x0000000008380002,0x0000002c14000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000300000001,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000040,0x000000000000004e,0x0000000000000000\n"
".quad 0x0000000000000001,0x0000000000000000,0x000000030000000b,0x0000000000000000\n"
".quad 0x0000000000000000,0x000000000000008e,0x000000000000004e,0x0000000000000000\n"
".quad 0x0000000000000001,0x0000000000000000,0x0000000200000013,0x0000000000000000\n"
".quad 0x0000000000000000,0x00000000000000e0,0x0000000000000030,0x0000000200000002\n"
".quad 0x0000000000000008,0x0000000000000018,0x0000000100000032,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000110,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000001,0x0000000000000000,0x7000000b0000003f,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000110,0x00000000000000d8,0x0000000000000000\n"
".quad 0x0000000000000008,0x0000000000000008,0x0000004001010002,0x0000000000000368\n"
".quad 0x0000000000000000,0x0000005000010007,0x0000000000000000,0x0000000000000011\n"
".quad 0x0000000000000000,0x0000000000000000,0x33010102464c457f,0x0000000000000007\n"
".quad 0x0000007300be0002,0x0000000000000000,0x0000000000000000,0x00000000000001e8\n"
".quad 0x0000004000500550,0x0001000600400000,0x7472747368732e00,0x747274732e006261\n"
".quad 0x746d79732e006261,0x746d79732e006261,0x78646e68735f6261,0x666e692e766e2e00\n"
".quad 0x67756265642e006f,0x2e00656d6172665f,0x612e6c65722e766e,0x2e00006e6f697463\n"
".quad 0x6261747274736873,0x6261747274732e00,0x6261746d79732e00,0x6261746d79732e00\n"
".quad 0x2e0078646e68735f,0x006f666e692e766e,0x665f67756265642e,0x766e2e00656d6172\n"
".quad 0x7463612e6c65722e,0x00000000006e6f69,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x000500030000003f,0x0000000000000000,0x0000000000000000\n"
".quad 0x000000000000004b,0x222f0a1008020200,0x0000000008000000,0x0000000008080000\n"
".quad 0x0000000008100000,0x0000000008180000,0x0000000008200000,0x0000000008280000\n"
".quad 0x0000000008300000,0x0000000008380000,0x0000000008000001,0x0000000008080001\n"
".quad 0x0000000008100001,0x0000000008180001,0x0000000008200001,0x0000000008280001\n"
".quad 0x0000000008300001,0x0000000008380001,0x0000000008000002,0x0000000008080002\n"
".quad 0x0000000008100002,0x0000000008180002,0x0000000008200002,0x0000000008280002\n"
".quad 0x0000000008300002,0x0000000008380002,0x0000002c14000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000300000001\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000040,0x000000000000004e\n"
".quad 0x0000000000000000,0x0000000000000001,0x0000000000000000,0x000000030000000b\n"
".quad 0x0000000000000000,0x0000000000000000,0x000000000000008e,0x000000000000004e\n"
".quad 0x0000000000000000,0x0000000000000001,0x0000000000000000,0x0000000200000013\n"
".quad 0x0000000000000000,0x0000000000000000,0x00000000000000e0,0x0000000000000030\n"
".quad 0x0000000200000002,0x0000000000000008,0x0000000000000018,0x0000000100000032\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000110,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000001,0x0000000000000000,0x7000000b0000003f\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000110,0x00000000000000d8\n"
".quad 0x0000000000000000, 0x0000000000000008, 0x0000000000000008\n"
".text\n");
#ifdef __cplusplus
extern "C" {
#endif
extern const unsigned long long fatbinData[671];
#ifdef __cplusplus
}
#endif
#ifdef __cplusplus
extern "C" {
#endif
static const __fatBinC_Wrapper_t __fatDeviceText __attribute__ ((aligned (8))) __attribute__ ((section (__CUDAFATBINSECTION)))= 
	{ 0x466243b1, 1, fatbinData, 0 };
#ifdef __cplusplus
}
#endif
