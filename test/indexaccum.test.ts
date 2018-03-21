import {Tensor, TensorMath} from '../src/index';

test('argmax vector', function() {
  let x = Tensor.create([1, 3, 2]);
  let z = TensorMath.argMax(x);
  let expected = Tensor.scalar(1);

  expect(z).toEqual(expected);
});

test('argmax vector - keep dims', function() {
  let x = Tensor.create([1, 3, 2]);
  let z = TensorMath.argMax(x, 0, true);
  let expected = Tensor.create([1]);

  expect(z).toEqual(expected);
});

test('argmin vector - keep dims', function() {
  let x = Tensor.create([3, 1, -1]);
  let z = TensorMath.argMin(x, 0, true);
  let expected = Tensor.create([2]);

  expect(z).toEqual(expected);
});

test('argmin vector', function() {
  let x = Tensor.create([3, 1, -1]);
  let z = TensorMath.argMin(x);
  let expected = Tensor.create(2);

  expect(z).toEqual(expected);
});