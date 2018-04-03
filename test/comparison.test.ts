import {Tensor} from "../src/index";

test("equal", function () {
  let x = Tensor.create([[1, 3, 3], [2, 3, 5]]);
  let y = Tensor.create([[2, 2, 4], [2, 3, 4]]);

  let z = x.equal(y);

  let expected = Tensor.create([[0, 0, 0], [1, 1, 0]]);

  expect(z).toEqual(expected);
});

test("not equal", function () {
  let x = Tensor.create([[1, 3, 3], [2, 3, 5]]);
  let y = Tensor.create([[2, 2, 4], [2, 3, 4]]);

  let z = x.notEqual(y);

  let expected = Tensor.create([[1, 1, 1], [0, 0, 1]]);

  expect(z).toEqual(expected);
});

test("greater", function () {
  let x = Tensor.create([[1, 3, 3], [2, 3, 5]]);
  let y = Tensor.create([[2, 2, 4], [2, 3, 4]]);

  let z = x.greater(y);

  let expected = Tensor.create([[0, 1, 0], [0, 0, 1]]);

  expect(z).toEqual(expected);
});

test("greaterEqual", function () {
  let x = Tensor.create([[1, 3, 3], [2, 3, 5]]);
  let y = Tensor.create([[2, 2, 4], [2, 3, 4]]);

  let z = x.greaterEqual(y);

  let expected = Tensor.create([[0, 1, 0], [1, 1, 1]]);

  expect(z).toEqual(expected);
});

test("less", function () {
  let x = Tensor.create([[1, 3, 3], [2, 3, 5]]);
  let y = Tensor.create([[2, 2, 4], [2, 3, 4]]);

  let z = x.less(y);

  let expected = Tensor.create([[1, 0, 1], [0, 0, 0]]);

  expect(z).toEqual(expected);
});


test("lessEqual", function () {
  let x = Tensor.create([[1, 3, 3], [2, 3, 5]]);
  let y = Tensor.create([[2, 2, 4], [2, 3, 4]]);

  let z = x.lessEqual(y);

  let expected = Tensor.create([[1, 0, 1], [1, 1, 0]]);

  expect(z).toEqual(expected);
});