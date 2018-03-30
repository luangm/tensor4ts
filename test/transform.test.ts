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

test('expm1', function () {
  let x = Tensor.linspace(1, 4, 4);
  let z = TensorMath.expm1(x);
  let expected = Tensor.create([Math.expm1(1), Math.expm1(2), Math.expm1(3), Math.expm1(4)]);
  expect(z).toEqual(expected);
});

test('log1p', function () {
  let x = Tensor.linspace(1, 4, 4);
  let z = TensorMath.log1p(x);
  let expected = Tensor.create([Math.log1p(1), Math.log1p(2), Math.log1p(3), Math.log1p(4)]);
  expect(z).toEqual(expected);
});

test('reciprocol', function () {
  let x = Tensor.linspace(1, 4, 4);
  let z = TensorMath.reciprocal(x);
  let expected = Tensor.create([1 / 1, 1 / 2, 1 / 3, 1 / 4]);
  expect(z).toEqual(expected);
});

test('sign', function () {
  let x = Tensor.create([-1, -2, 1, 2, 0]);
  let z = TensorMath.sign(x);
  let expected = Tensor.create([-1, -1, 1, 1, 0]);
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

test('step', function () {
  let x = Tensor.create([-1, -2, 1, 2]);
  let z = TensorMath.step(x);
  let expected = Tensor.create([0, 0, 1, 1]);
  expect(z).toEqual(expected);
});

test('square', function () {
  let x = Tensor.create([-1, -2, 1, 2]);
  let z = TensorMath.square(x);
  let expected = Tensor.create([1, 4, 1, 4]);
  expect(z).toEqual(expected);
});

test('power 2', function () {
  let x = Tensor.create([-1, -2, 1, 2]);
  let z = TensorMath.pow(x, 2);
  let expected = Tensor.create([1, 4, 1, 4]);
  expect(z).toEqual(expected);
});

test('power 3', function () {
  let x = Tensor.create([-1, -2, 1, 2]);
  let z = TensorMath.pow(x, 3);
  let expected = Tensor.create([-1, -8, 1, 8]);
  expect(z).toEqual(expected);
});

test('floor', function () {
  let x = Tensor.create([-1.1, -2.1, 1.1, 2.1]);
  let z = TensorMath.floor(x);
  let expected = Tensor.create([-2, -3, 1, 2]);
  expect(z).toEqual(expected);
});


test('ceil', function () {
  let x = Tensor.create([-1.1, -2.1, 1.1, 2.1]);
  let z = TensorMath.ceil(x);
  let expected = Tensor.create([-1, -2, 2, 3]);
  expect(z).toEqual(expected);
});

test('round', function () {
  let x = Tensor.create([-1.1, -2.1, 1.1, 2.1]);
  let z = TensorMath.round(x);
  let expected = Tensor.create([-1, -2, 1, 2]);
  expect(z).toEqual(expected);
});

test('asin', function () {
  let x = Tensor.create([1, 2, 3]);
  let z = TensorMath.asin(x);
  let expected = Tensor.create([Math.asin(1), Math.asin(2), Math.asin(3)]);
  expect(z).toEqual(expected);
});

test('acos', function () {
  let x = Tensor.create([1, 2, 3]);
  let z = TensorMath.acos(x);
  let expected = Tensor.create([Math.acos(1), Math.acos(2), Math.acos(3)]);
  expect(z).toEqual(expected);
});

test('atan', function () {
  let x = Tensor.create([1, 2, 3]);
  let z = TensorMath.atan(x);
  let expected = Tensor.create([Math.atan(1), Math.atan(2), Math.atan(3)]);
  expect(z).toEqual(expected);
});

test('tanh', function () {
  let x = Tensor.create([-1, -2, 1, 2]);
  let z = TensorMath.tanh(x);
  let expected = Tensor.create([Math.tanh(-1), Math.tanh(-2), Math.tanh(1), Math.tanh(2)]);
  expect(z).toEqual(expected);
});

test('sigmoid', function () {
  let x = Tensor.create([1, 2, 3]);
  let z = TensorMath.sigmoid(x);
  let expected = Tensor.create([1 / (1 + Math.exp(-1)), 1 / (1 + Math.exp(-2)), 1 / (1 + Math.exp(-3))]);
  expect(z).toEqual(expected);
});

test('softplus', function () {
  let x = Tensor.create([[1, 2, 3], [4, 5, 6]]);
  let z = TensorMath.softplus(x);
  let expected = Tensor.create([
    [softplus(1), softplus(2), softplus(3)],
    [softplus(4), softplus(5), softplus(6)]
  ]);
  expect(z).toEqual(expected);
  console.log(z.toString());
});

function softplus(x: number): number {
  return Math.log1p(Math.exp(x));
}

test('softmax - vector', function () {
  let x = Tensor.linspace(1, 3, 3);
  let z = TensorMath.softmax(x);
  let sum = Math.exp(1) + Math.exp(2) + Math.exp(3);
  let expected = Tensor.create([Math.exp(1) / sum, Math.exp(2) / sum, Math.exp(3) / sum]);
  expect(z).toEqual(expected);
});

test('softmax - matrix', function () {
  let x = Tensor.linspace(1, 6, 6).reshape([2, 3]);
  let z = TensorMath.softmax(x);
  let sum0 = Math.exp(1) + Math.exp(2) + Math.exp(3);
  let sum1 = Math.exp(4) + Math.exp(5) + Math.exp(6);

  let expected = Tensor.create(
      [[Math.exp(1) / sum0, Math.exp(2) / sum0, Math.exp(3) / sum0],
        [Math.exp(4) / sum1, Math.exp(5) / sum1, Math.exp(6) / sum1]]
  );
  expect(z).toEqual(expected);
});

test('softmax - matrix 0', function () {
  let x = Tensor.linspace(1, 6, 6).reshape([2, 3]);
  let z = TensorMath.softmax(x, 0);

  let sum0 = Math.exp(1) + Math.exp(4);
  let sum1 = Math.exp(2) + Math.exp(5);
  let sum2 = Math.exp(3) + Math.exp(6);
  //
  let expected = Tensor.create(
      [[Math.exp(1) / sum0, Math.exp(2) / sum1, Math.exp(3) / sum2],
        [Math.exp(4) / sum0, Math.exp(5) / sum1, Math.exp(6) / sum2]]
  );
  expect(z).toEqual(expected);
});