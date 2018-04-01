import {Tensor, TensorMath} from '../src/index';

test('tile vector', function () {
  let x = Tensor.create([1, 2, 3]);
  let z = TensorMath.tile(x, [2]);
  let expected = Tensor.create([1, 2, 3, 1, 2, 3]);
  expect(z).toEqual(expected);
});

test('tile 2', function () {
  let x = Tensor.create([[1, 2, 3]]);
  let z = TensorMath.tile(x, [2, 2]);
  console.log(z.toString());
  let expected = Tensor.create([[1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3]]);
  expect(z).toEqual(expected);
});

test('tile 2 1', function () {
  let x = Tensor.create([[1, 2, 3]]);
  let z = TensorMath.tile(x, [2, 1]);
  console.log(z.toString());
  let expected = Tensor.create([[1, 2, 3,], [1, 2, 3]]);
  expect(z).toEqual(expected);
});

test('tile 3d 1', function () {
  let x = Tensor.create([[[1]]]);
  let z = TensorMath.tile(x, [2, 3, 4]);
  console.log(z.toString());
  let expected = Tensor.create([
    [[1, 1, 1, 1],
      [1, 1, 1, 1],
      [1, 1, 1, 1]],

    [[1, 1, 1, 1],
      [1, 1, 1, 1],
      [1, 1, 1, 1]]
  ]);
  expect(z).toEqual(expected);
});