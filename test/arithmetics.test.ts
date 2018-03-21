import {Tensor, TensorMath} from '../src/index';

test('add', function () {
  let x = Tensor.create([[1, 2, 3], [4, 5, 6]]);
  let y = Tensor.create([[2, 3, 4], [5, 6, 7]]);
  let z = x.add(y);
  let expected = Tensor.create([[3, 5, 7], [9, 11, 13]]);

  expect(z).toEqual(expected);
  expect(z).not.toBe(x);
});

test('addi', function () {
  let x = Tensor.create([[1, 2, 3], [4, 5, 6]]);
  let y = Tensor.create([[2, 3, 4], [5, 6, 7]]);
  let z = x.addi(y);
  let expected = Tensor.create([[3, 5, 7], [9, 11, 13]]);

  expect(z).toEqual(expected);
  expect(z).toBe(x);
});

test('addi - broadcast', function () {
  let x = Tensor.create([[1, 2, 3], [4, 5, 6]]);
  let y = Tensor.create([[2], [3]]);
  let z = x.addi(y);
  let expected = Tensor.create([[3, 4, 5], [7, 8, 9]]);

  expect(z).toEqual(expected);
  expect(z).toBe(x);
});

test('add - broadcast', function () {
  let x = Tensor.create([[1, 2, 3]]);
  let y = Tensor.create([[2], [3]]);
  let z = x.add(y);
  let expected = Tensor.create([[3, 4, 5], [4, 5, 6]]);

  expect(z).toEqual(expected);
});

test('add vector', function () {
  let x = Tensor.create([1, 2, 3]);
  let y = Tensor.create([4, 5, 6]);
  let z = x.add(y);
  let expected = Tensor.create([5, 7, 9]);

  expect(z).toEqual(expected);
});

test('scalar add', function () {
  let x = Tensor.create(3);
  let y = Tensor.create(5);
  let z = x.add(y);
  let expected = Tensor.create(8);

  expect(z).toEqual(expected);
});

test('vector add scalar', function () {
  let x = Tensor.create([1, 2, 3]);
  let y = Tensor.create(5);
  let z = x.add(y);
  let expected = Tensor.create([6, 7, 8]);

  expect(z).toEqual(expected);
});

test('matrix add scalar', function () {
  let x = Tensor.create([[1, 2, 3], [4, 5, 6]]);
  let y = Tensor.create(5);
  let z = x.add(y);
  let expected = Tensor.create([[6, 7, 8], [9, 10, 11]]);

  expect(z).toEqual(expected);
});

test('matrix add vector', function () {
  let x = Tensor.create([[1, 2, 3], [4, 5, 6]]);
  let y = Tensor.create([1, 2, 3]);
  let z = x.add(y);
  let expected = Tensor.create([[2, 4, 6], [5, 7, 9]]);

  expect(z).toEqual(expected);
});

test('matrix add vector2', function () {
  let x = Tensor.create([[1, 2]]); // 1x2
  let y = Tensor.create([2, 3]); // 2
  let z = x.add(y);
  let expected = Tensor.create([[3, 5]]);

  expect(z).toEqual(expected);
});

test('3d add scalar', function () {
  let x = Tensor.create([[[1, 2, 3], [4, 5, 6]]]);
  let y = Tensor.create(2);
  let z = x.add(y);
  let expected = Tensor.create([[[3, 4, 5], [6, 7, 8]]]);

  expect(z).toEqual(expected);
});

test('3d add vector', function () {
  let x = Tensor.create([[[1, 2, 3], [4, 5, 6]]]);
  let y = Tensor.create([1, 2, 3]);
  let z = x.add(y);
  let expected = Tensor.create([[[2, 4, 6], [5, 7, 9]]]);

  expect(z).toEqual(expected);
});

// TODO: BUG - This SHOULD throw error since not possible.
test('addi - throw error', function () {
  let x = Tensor.create([[1, 2, 3]]);
  let y = Tensor.create([[2], [3]]);
  let z = x.addi(y);

  throw new Error();
});

test('subtract', function () {
  let x = Tensor.create([1, 2, 3, 4, 5, 6]).reshape([2, 3]);
  let y = Tensor.create([2, 4, 6, 3, 1, 2]).reshape([2, 3]);
  let z = x.subtract(y);
  let expected = Tensor.create([[-1, -2, -3], [1, 4, 4]]);

  expect(z).toEqual(expected);
});

test('multiply', function () {
  let x = Tensor.create([1, 2, 3, 4, 5, 6]).reshape([2, 3]);
  let y = Tensor.create([2, 4, 6, 3, 1, 2]).reshape([2, 3]);
  let z = x.multiply(y);
  let expected = Tensor.create([[2, 8, 18], [12, 5, 12]]);

  expect(z).toEqual(expected);
});

test('divide', function () {
  let x = Tensor.create([1, 2, 3, 4, 5, 6]).reshape([2, 3]);
  let y = Tensor.create([2, 4, 6, 3, 1, 2]).reshape([2, 3]);
  let z = x.divide(y);
  let expected = Tensor.create([[1 / 2, 2 / 4, 3 / 6], [4 / 3, 5, 6 / 2]]);

  expect(z).toEqual(expected);
});

test('max', function () {
  let x = Tensor.create([1, 2, 3, 4, 5, 6]).reshape([2, 3]);
  let y = Tensor.create([2, 4, 6, 3, 1, 2]).reshape([2, 3]);
  let z = TensorMath.max(x, y);
  let expected = Tensor.create([[2, 4, 6], [4, 5, 6]]);

  expect(z).toEqual(expected);
});

test('max - with broadcast', function () {
  let x = Tensor.create([1, 2, 3]).reshape([1, 3]);
  let y = Tensor.create([2, 4]).reshape([2, 1]);
  let z = TensorMath.max(x, y);
  let expected = Tensor.create([[2, 2, 3], [4, 4, 4]]);

  expect(z).toEqual(expected);
});

test('min - with broadcast', function () {
  let x = Tensor.create([1, 2, 3]).reshape([1, 3]);
  let y = Tensor.create([2, 4]).reshape([2, 1]);
  let z = TensorMath.min(x, y);
  let expected = Tensor.create([[1, 2, 2], [1, 2, 3]]);

  expect(z).toEqual(expected);
});

test('mod - with broadcast', function () {
  let x = Tensor.create([5, 6, 7]).reshape([1, 3]);
  let y = Tensor.create([2, 4]).reshape([2, 1]);
  let z = x.mod(y);
  let expected = Tensor.create([[1, 0, 1], [1, 2, 3]]);

  expect(z).toEqual(expected);
});