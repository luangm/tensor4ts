import {Tensor} from "../src/index";

test("transpose no arg 2d", function () {
  let x = Tensor.create([[1, 2, 3], [4, 5, 6]]);
  let z = x.transpose();
  console.log(z.toString());

  expect(z.shape).toEqual([3, 2]);
});

test("transpose no arg 3d", function () {
  let x = Tensor.create([[[1, 2, 3], [4, 5, 6]]]);
  let z = x.transpose();
  console.log(z.toString());

  expect(z.shape).toEqual([3, 2, 1]);
});

test("transpose 3d", function () {
  let x = Tensor.create([[[1, 2, 3], [4, 5, 6]]]);
  let z = x.transpose([2, 0, 1]);
  console.log(z.toString());

  expect(z.shape).toEqual([3, 1, 2]);
});

test("transpose 3d 2", function () {
  let x = Tensor.create([[[1, 2, 3], [4, 5, 6]]]);
  expect(() => {
    let z = x.transpose([0, 1]);
  }).toThrow();
});

test("transpose no arg 2d with dup", function () {
  let x = Tensor.create([[1, 2, 3], [4, 5, 6]]);
  let z = x.transpose().dup();
  console.log(z.toString());

  expect(z.shape).toEqual([3, 2]);

  let expected = Tensor.create([[1, 4], [2, 5], [3, 6]]);
  expect(z).toEqual(expected);
  console.log(z);
});