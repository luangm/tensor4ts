var sum = new Function("a", "b", "return a + b");

console.log(sum(2, 6));
console.log(sum.length);
console.log(sum.name);


//
// var s0 = SS[0], s1 = SS[1], t0p0 = t0[0], t0p1 = t0[1], t1p0 = t1[0], t1p1 = t1[1], t2p0 = t2[0], t2p1 = t2[1];
// let p0 = 0;
// let p1 = 0;
// let p2 = 0;
// var offset0 = p0;
// var offset1 = p1;
// var offset2 = p2;
// for (var j0 = SS[1] | 0; j0 > 0;) {
//   if (j0 < 64) {
//     s1 = j0;
//     j0 = 0;
//   } else {
//     s1 = 64;
//     j0 -= 64;
//   }
//   for (var j1 = SS[0] | 0; j1 > 0;) {
//     if (j1 < 64) {
//       s0 = j1;
//       j1 = 0;
//     } else {
//       s0 = 64;
//       j1 -= 64;
//     }
//     p0 = (offset0 + j0 * t0p1 + j1 * t0p0);
//     p1 = (offset1 + j0 * t1p1 + j1 * t1p0);
//     p2 = (offset2 + j0 * t2p1 + j1 * t2p0);
//     var i0 = 0, i1 = 0, d0s0 = t0p1, d0s1 = (t0p0 - s1 * t0p1), d1s0 = t1p1, d1s1 = (t1p0 - s1 * t1p1), d2s0 = t2p1,
//       d2s1 = (t2p0 - s1 * t2p1);
//     for (i1 = 0; i1 < s0; ++i1) {
//       for (i0 = 0; i0 < s1; ++i0) {
//         {
//           a2[p2] = a0[p0] + a1[p1];
//         }
//         p0 += d0s0;
//         p1 += d1s0;
//         p2 += d2s0;
//       }
//       p0 += d0s1;
//       p1 += d1s1;
//       p2 += d2s1;
//     }
//   }
// }
//
// var s0 = SS[0], s1 = SS[1], t0p0 = t0[0], t0p1 = t0[1], t1p0 = t1[0], t1p1 = t1[1], t2p0 = t2[0], t2p1 = t2[1];
// p0 |= 0;
// p1 |= 0;
// p2 |= 0;
// var offset0 = p0;
// var offset1 = p1;
// var offset2 = p2;
// for (var j0 = SS[1] | 0; j0 > 0;) {
//   if (j0 < 64) {
//     s1 = j0;
//     j0 = 0;
//   } else {
//     s1 = 64;
//     j0 -= 64;
//   }
//   for (var j1 = SS[0] | 0; j1 > 0;) {
//     if (j1 < 64) {
//       s0 = j1;
//       j1 = 0;
//     } else {
//       s0 = 64;
//       j1 -= 64;
//     }
//     p0 = (offset0 + j0 * t0p1 + j1 * t0p0);
//     p1 = (offset1 + j0 * t1p1 + j1 * t1p0);
//     p2 = (offset2 + j0 * t2p1 + j1 * t2p0);
//     var i0 = 0, i1 = 0, d0s0 = t0p1, d0s1 = (t0p0 - s1 * t0p1), d1s0 = t1p1, d1s1 = (t1p0 - s1 * t1p1), d2s0 = t2p1,
//       d2s1 = (t2p0 - s1 * t2p1);
//     for (i1 = 0; i1 < s0; ++i1) {
//       for (i0 = 0; i0 < s1; ++i0) {
//         {
//           a2[p2] = a0[p0] + a1[p1];
//         }
//         p0 += d0s0;
//         p1 += d1s0;
//         p2 += d2s0;
//       }
//       p0 += d0s1;
//       p1 += d1s1;
//       p2 += d2s1;
//     }
//   }
// }
//
// "use strict";
// var s0 = SS[0], s1 = SS[1], s2 = SS[2];
// p0 |= 0;
// p1 |= 0;
// var offset0 = p0;
// var offset1 = p1;
// for (var j0 = SS[1] | 0; j0 > 0;) {
//   if (j0 < 64) {
//     s1 = j0;
//     j0 = 0;
//   } else {
//     s1 = 64;
//     j0 -= 64;
//   }
//   for (var j1 = SS[0] | 0; j1 > 0;) {
//     if (j1 < 64) {
//       s0 = j1;
//       j1 = 0;
//     } else {
//       s0 = 64;
//       j1 -= 64;
//     }
//     for (var j2 = SS[2] | 0; j2 > 0;) {
//       if (j2 < 64) {
//         s2 = j2;
//         j2 = 0;
//       } else {
//         s2 = 64;
//         j2 -= 64;
//       }
//       p0 = (offset0 + j0 * t0[1] + j1 * t0[0] + j2 * t0[2]);
//       p1 = (offset1 + j0 * t1[1] + j1 * t1[0] + j2 * t1[2]);
//       var i0 = 0, i1 = 0, i2 = 0, d0s0 = t0[1], d0s1 = (t0[0] - s1 * t0[1]), d0s2 = (t0[2] - s0 * t0[0]), d1s0 = t1[1],
//         d1s1 = (t1[0] - s1 * t1[1]), d1s2 = (t1[2] - s0 * t1[0]);
//       for (i2 = 0; i2 < s2; ++i2) {
//         for (i1 = 0; i1 < s0; ++i1) {
//           for (i0 = 0; i0 < s1; ++i0) {
//             var l0 = a0[p0];
//             var l1 = a1[p1];
//             {
//               l0 += l1 + 0.1;
//               l1 -= l0 * 0.5;
//             }
//             a0[p0] = l0;
//             a1[p1] = l1;
//             p0 += d0s0;
//             p1 += d1s0;
//           }
//           p0 += d0s1;
//           p1 += d1s1;
//         }
//         p0 += d0s2;
//         p1 += d1s2;
//       }
//     }
//   }
// }
