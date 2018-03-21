import {Tensor, TensorMath} from '../src/index';

test('exp scalar', function () {
  let x = Tensor.scalar(5);
  let z = TensorMath.exp(x);
  let expected = Tensor.scalar(Math.exp(5));
  expect(z).toEqual(expected);
});

test('exp vector', function () {
  let x = Tensor.linspace(1, 4, 4);
  let z = TensorMath.exp(x);
  let expected = Tensor.create([Math.exp(1), Math.exp(2), Math.exp(3), Math.exp(4)]);
  expect(z).toEqual(expected);
});

test('exp matrix', function () {
  let x = Tensor.linspace(1, 4, 4).reshape([2, 2]);
  let z = TensorMath.exp(x);
  let expected = Tensor.create([[Math.exp(1), Math.exp(2)], [Math.exp(3), Math.exp(4)]]);
  expect(z).toEqual(expected);
});

test('log', function () {
  let x = Tensor.linspace(1, 4, 4);
  let z = TensorMath.log(x);
  let expected = Tensor.create([Math.log(1), Math.log(2), Math.log(3), Math.log(4)]);
  expect(z).toEqual(expected);
});

test('abs', function () {
  let x = Tensor.create([-1, 1, -2, 3]);
  let z = TensorMath.abs(x);
  let expected = Tensor.create([1, 1, 2, 3]);
  expect(z).toEqual(expected);
});

test('abs 3D', function () {
  let x = Tensor.create([[[-1, -2], [3, 4], [5, -6]], [[-1, -3], [-3, 4], [5, -6]]]);
  let z = TensorMath.abs(x);
  let expected = Tensor.create([[[1, 2], [3, 4], [5, 6]], [[1, 3], [3, 4], [5, 6]]]);
  expect(z).toEqual(expected);
});

test('negate', function () {
  let x = Tensor.create([-1, 1, -2, 3]);
  let z = x.negate();
  let expected = Tensor.create([1, -1, 2, -3]);
  expect(z).toEqual(expected);
});

test('cos', function () {
  let x = Tensor.linspace(1, 4, 4);
  let z = TensorMath.cos(x);
  let expected = Tensor.create([Math.cos(1), Math.cos(2), Math.cos(3), Math.cos(4)]);
  expect(z).toEqual(expected);
});

test('cosh', function () {
  let x = Tensor.linspace(1, 4, 4);
  let z = TensorMath.cosh(x);
  let expected = Tensor.create([Math.cosh(1), Math.cosh(2), Math.cosh(3), Math.cosh(4)]);
  expect(z).toEqual(expected);
});

test('acos', function () {
  let x = Tensor.linspace(1, 4, 4);
  let z = TensorMath.acos(x);
  let expected = Tensor.create([Math.acos(1), Math.acos(2), Math.acos(3), Math.acos(4)]);
  expect(z).toEqual(expected);
});

test('acosh', function () {
  let x = Tensor.linspace(1, 4, 4);
  let z = TensorMath.acosh(x);
  let expected = Tensor.create([Math.acosh(1), Math.acosh(2), Math.acosh(3), Math.acosh(4)]);
  expect(z).toEqual(expected);
});

test('fill', function () {
  let x = Tensor.zeros([2, 2]);
  let z = TensorMath.fill(x, 5);
  let expected = Tensor.create([[5, 5], [5, 5]]);
  expect(z).toEqual(expected);
});

test('relu', function () {
  let x = Tensor.create([-1, -2, 1, 2]);
  let z = TensorMath.relu(x);
  let expected = Tensor.create([0, 0, 1, 2]);
  expect(z).toEqual(expected);
});

test('round', function () {
  let x = Tensor.create([-1.1, -2.1, 1.1, 2.1]);
  let z = TensorMath.round(x);
  let expected = Tensor.create([-1, -2, 1, 2]);
  expect(z).toEqual(expected);
});

test('sigmoid', function () {
  let x = Tensor.create([1, 2, 3]);
  let z = TensorMath.sigmoid(x);
  let expected = Tensor.create([1 / (1 + Math.exp(-1)), 1 / (1 + Math.exp(-2)), 1 / (1 + Math.exp(-3))]);
  expect(z).toEqual(expected);
});