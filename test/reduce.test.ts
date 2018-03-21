import {Tensor, TensorMath} from '../src/index';

test('reduceSum vector', function() {
  let x = Tensor.create([1, 2, 3]);
  let z = TensorMath.reduceSum(x);
  let expected = Tensor.scalar(6);

  expect(z).toEqual(expected);
});

test('reduceSum matrix all', function() {
  let x = Tensor.create([[1, 2, 3], [4, 5, 6]]);
  let z = TensorMath.reduceSum(x);
  let expected = Tensor.scalar(21);

  expect(z).toEqual(expected);
});

test('reduceSum matrix all - keepDims', function() {
  let x = Tensor.create([[1, 2, 3], [4, 5, 6]]);
  let z = TensorMath.reduceSum(x, -1, true);
  let expected = Tensor.create([[21]]);

  expect(z).toEqual(expected);
});

test('reduceSum 3d all', function() {
  let x = Tensor.create([[[1, 2, 3], [4, 5, 6]]]);
  let z = TensorMath.reduceSum(x);
  let expected = Tensor.scalar(21);

  expect(z).toEqual(expected);
});

test('reduceSum 3d 0', function() {
  let x = Tensor.create([[[1, 2, 3], [4, 5, 6]]]);
  let z = TensorMath.reduceSum(x, 0);
  let expected = Tensor.create([[1, 2, 3], [4, 5, 6]]);

  expect(z).toEqual(expected);
});

test('reduceSum 3d 1', function() {
  let x = Tensor.create([[[1, 2, 3], [4, 5, 6]]]);
  let z = TensorMath.reduceSum(x, 1);
  let expected = Tensor.create([[5, 7, 9]]);

  expect(z).toEqual(expected);
});

test('reduceSum 3d 2', function() {
  let x = Tensor.create([[[1, 2, 3], [4, 5, 6]]]);
  let z = TensorMath.reduceSum(x, 2);
  let expected = Tensor.create([[6, 15]]);

  expect(z).toEqual(expected);
});

test('reduceSum 3d 1 2', function() {
  let x = Tensor.create([[[1, 2, 3], [4, 5, 6]]]);
  let z = TensorMath.reduceSum(x, [1, 2]);
  let expected = Tensor.create([21]);

  expect(z).toEqual(expected);
});

test('reduceSum 3d 0+1', function() {
  let x = Tensor.create([[[1, 2, 3], [4, 5, 6]]]);
  let z = TensorMath.reduceSum(x, [0, 1]);
  let expected = Tensor.create([5, 7, 9]);

  expect(z).toEqual(expected);
});

test('reduceSum 3d 0+1 - keep dims', function() {
  let x = Tensor.create([[[1, 2, 3], [4, 5, 6]]]);
  let z = TensorMath.reduceSum(x, [0, 1], true);
  let expected = Tensor.create([[[5, 7, 9]]]);

  expect(z).toEqual(expected);
});

test('reduceSum 0', function() {
  let x = Tensor.create([[1, 2, 3], [4, 5, 6]]);
  let z = TensorMath.reduceSum(x, 0);
  let expected = Tensor.create([5, 7, 9]);

  expect(z).toEqual(expected);
});

test('reduceSum 0 - keepDims', function() {
  let x = Tensor.create([[1, 2, 3], [4, 5, 6]]);
  let z = TensorMath.reduceSum(x, 0, true);
  let expected = Tensor.create([[5, 7, 9]]);

  expect(z).toEqual(expected);
});

test('reduceSum 1', function() {
  let x = Tensor.create([[1, 2, 3], [4, 5, 6]]);
  let z = TensorMath.reduceSum(x, 1);
  let expected = Tensor.create([6, 15]);

  expect(z).toEqual(expected);
});

test('reduceSum 1 - keepDims', function() {
  let x = Tensor.create([[1, 2, 3], [4, 5, 6]]);
  let z = TensorMath.reduceSum(x, 1, true);
  let expected = Tensor.create([[6], [15]]);

  expect(z).toEqual(expected);
});

test('reduceMax vector', function() {
  let x = Tensor.create([1, 3, 2]);
  let z = TensorMath.reduceMax(x);
  let expected = Tensor.scalar(3);

  expect(z).toEqual(expected);
});


test('reduceMax vector - neg case', function() {
  let x = Tensor.create([-1, -3, -2]);
  let z = TensorMath.reduceMax(x);
  let expected = Tensor.scalar(-1);

  expect(z).toEqual(expected);
});

test('reduceMin vector', function() {
  let x = Tensor.create([1, 3, 2]);
  let z = TensorMath.reduceMin(x);
  let expected = Tensor.scalar(1);

  expect(z).toEqual(expected);
});

test('reduceMean vector', function() {
  let x = Tensor.create([1, 3, 2]);
  let z = TensorMath.reduceMean(x);
  let expected = Tensor.scalar(2);

  expect(z).toEqual(expected);
});


test('reduceMean matrix all', function() {
  let x = Tensor.create([[1, 2, 3], [4, 5, 6]]);
  let z = TensorMath.reduceMean(x);
  let expected = Tensor.scalar((1+2+3+4+5+6)/6);

  expect(z).toEqual(expected);
});

test('reduceMax matrix all', function() {
  let x = Tensor.create([[1, 2, 3], [4, 5, 6]]);
  let z = TensorMath.reduceMax(x);
  let expected = Tensor.scalar(6);

  expect(z).toEqual(expected);
});

test('reduceMin matrix all', function() {
  let x = Tensor.create([[1, 2, 3], [4, 5, 6]]);
  let z = TensorMath.reduceMin(x);
  let expected = Tensor.scalar(1);

  expect(z).toEqual(expected);
});

test('reduceMax 3d all keepDims', function() {
  let x = Tensor.create([[[1, 2, 3], [4, 5, 6]]]);
  let z = TensorMath.reduceMax(x, -1, true);
  let expected = Tensor.create([[[6]]]);

  expect(z).toEqual(expected);
});

test('reduceMin 3d 1', function() {
  let x = Tensor.create([[[1, 2, 3], [4, 5, 6]]]);
  let z = TensorMath.reduceMin(x, 1);
  let expected = Tensor.create([[1, 2, 3]]);

  expect(z).toEqual(expected);
});

test('reduceMean 3d 1', function() {
  let x = Tensor.create([[[1, 2, 3], [4, 5, 6]]]);
  let z = TensorMath.reduceMean(x, 1);
  let expected = Tensor.create([[5/2, 7/2, 9/2]]);

  expect(z).toEqual(expected);
});

test('reduceMax 3d 1', function() {
  let x = Tensor.create([[[1, 2, 3], [4, 5, 6]]]);
  let z = TensorMath.reduceMax(x, 1);
  let expected = Tensor.create([[4, 5, 6]]);

  expect(z).toEqual(expected);
});

test('reduceMax 3d 2', function() {
  let x = Tensor.create([[[1, 2, 3], [4, 5, 6]]]);
  let z = TensorMath.reduceMax(x, 2);
  let expected = Tensor.create([[3, 6]]);

  expect(z).toEqual(expected);
});

test('reduceMin 3d 0', function() {
  let x = Tensor.create([[[1, 2, 3], [4, 5, 6]]]);
  let z = TensorMath.reduceMin(x, 0);
  let expected = Tensor.create([[1, 2, 3], [4, 5, 6]]);

  expect(z).toEqual(expected);
});

test('reduceProd 3d 0', function() {
  let x = Tensor.create([[[1, 2, 3], [4, 5, 6]]]);
  let z = TensorMath.reduceProd(x, 0);
  let expected = Tensor.create([[1, 2, 3], [4, 5, 6]]);

  expect(z).toEqual(expected);
});

test('reduceProd 3d all', function() {
  let x = Tensor.create([[[1, 2, 3], [4, 5, 6]]]);
  let z = TensorMath.reduceProd(x);
  let expected = Tensor.create(1*2*3*4*5*6);

  expect(z).toEqual(expected);
});