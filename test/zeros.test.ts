import {Tensor, TensorMath} from "../src/index";

test("add vector", function () {
  let x = Tensor.create([1, 2, 3]);
  let y = Tensor.sparseZeros([3]);

  let z = x.add(y);
  let expected = Tensor.create([1, 2, 3]);

  expect(z).toEqual(expected);
  // console.log(z.toString());
});


test("add scalar", function () {
  let x = Tensor.create(5);
  let y = Tensor.sparseZeros([]);

  let z = x.add(y);
  let expected = Tensor.create(5);

  expect(z).toEqual(expected);
  // console.log(z.toString());
});

test("add matrix", function () {
  let x = Tensor.create([[1, 2, 3], [4, 5, 6]]);
  let y = Tensor.sparseZeros([2, 3]);

  let z = x.add(y);
  let expected = Tensor.create([[1, 2, 3], [4, 5, 6]]);

  expect(z).toEqual(expected);
  // console.log(z.toString());
});

test("multiply matrix", function () {
  let x = Tensor.create([[1, 2, 3], [4, 5, 6]]);
  let y = Tensor.sparseZeros([2, 3]);

  let z = x.multiply(y);
  let expected = Tensor.create([[0, 0, 0], [0, 0, 0]]);

  expect(z).toEqual(expected);
  // console.log(z.toString());
});


test("add 3d", function () {
  let x = Tensor.create([[[1, 2, 3], [4, 5, 6]], [[2, 3, 4], [5, 6, 7]]]);
  let y = Tensor.sparseZeros([2, 3]);

  let z = x.add(y);
  let expected = Tensor.create([[[1, 2, 3], [4, 5, 6]], [[2, 3, 4], [5, 6, 7]]]);

  expect(z).toEqual(expected);
  // console.log(z.toString());
});

test("abs matrix", function () {
  // let x = Tensor.create([[[1, 2, 3], [4, 5, 6]], [[2, 3, 4], [5, 6, 7]]]);
  let y = Tensor.sparseZeros([2, 3]);

  let z = y.abs();
  let expected = Tensor.create([[0, 0, 0], [0, 0, 0]]);

  expect(z).toEqual(expected);
  // console.log(z.toString());
});

test("exp matrix", function () {
  // let x = Tensor.create([[[1, 2, 3], [4, 5, 6]], [[2, 3, 4], [5, 6, 7]]]);
  let y = Tensor.sparseZeros([2, 3]);

  let z = y.exp();
  let expected = Tensor.create([[1, 1, 1], [1, 1, 1]]);

  expect(z).toEqual(expected);
  // console.log(z.toString());
});

test("print", function () {
  // let x = Tensor.create([[[1, 2, 3], [4, 5, 6]], [[2, 3, 4], [5, 6, 7]]]);
  let y = Tensor.sparseZeros([2, 3]);

  console.log(y.toString());
  // console.log(z.toString());
});

test("reshape", function () {
  // let x = Tensor.create([[[1, 2, 3], [4, 5, 6]], [[2, 3, 4], [5, 6, 7]]]);
  let y = Tensor.sparseZeros([1, 6]);

  console.log(y.toString());

  let z = y.reshape([2, 3]);

  console.log(z.toString());
});


test("broadcast", function () {
  // let x = Tensor.create([[[1, 2, 3], [4, 5, 6]], [[2, 3, 4], [5, 6, 7]]]);
  let y = Tensor.sparseZeros([1, 3]);

  console.log(y.toString());

  let z = y.broadcast([3,3]);

  console.log(z.toString());
  console.log(z);
});