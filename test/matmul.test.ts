import {Tensor} from "../src/index";

test("mm", function () {
  let x = Tensor.create([[1, 2, 3], [2, 3, 4]]); // 2x3
  let y = Tensor.create([[2, 3], [3, 4], [1, 1]]); // 3x2
  let z = x.matmul(y);
  let expected = Tensor.create([[11, 14], [4 + 9 + 4, 6 + 12 + 4]]);

  expect(z).toEqual(expected);
});

test("mm t f", function () {
  let x = Tensor.create([[1, 1, 1, 1]]); // 1x4
  let y = Tensor.create([[1, 2, 3, 4]]); // 1x4
  let z = x.matmul(y, true, false);
  let expected = Tensor.create([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]);
  // console.log(z.toString());
  expect(z).toEqual(expected);
});
