import {Tensor} from "../src/index";

test("slice + dup", function () {
  let x = Tensor.create([[1, 2, 3], [4, 5, 6]]);
  let z = x.slice([0, 1]).dup();
  let expected = Tensor.create([[2, 3], [5, 6]]);
  expect(z).toEqual(expected);
});

test("slice + mul", function () {
  let x = Tensor.create([[1, 2, 3], [4, 5, 6]]);
  let y = Tensor.create([5, 6, 7, 8]).reshape([2, 2]);
  let z = x.slice([0, 1]).multiply(y).dup();
  let expected = Tensor.create([[10, 18], [35, 48]]);
  expect(z).toEqual(expected);
});

test("slice + reduceSum", function () {
  let x = Tensor.create([[1, 2, 3], [4, 5, 6]]);
  // let y = Tensor.create([5, 6, 7, 8]).reshape([2, 2]);
  let z = x.slice([0, 1]).reduceSum(0);
  let expected = Tensor.create([7, 9]);
  expect(z).toEqual(expected);
});