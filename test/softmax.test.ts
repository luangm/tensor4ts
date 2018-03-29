import {Tensor, TensorMath} from '../src/index';

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