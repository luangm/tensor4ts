import {Tensor, TensorMath} from '../src/index';

test('repeat vector', function () {
  let x = Tensor.create([1, 2, 3]);
  let z = TensorMath.repeat(x, 2, 0);
  let expected = Tensor.create([1, 1, 2, 2, 3, 3]);
  expect(z).toEqual(expected);
});

test('repeat matrix', function () {
  let x = Tensor.create([[1, 2, 3]]);
  let z = TensorMath.repeat(x, 2, 0);
  // console.log(z.toString());
  let expected = Tensor.create([[1, 2, 3], [1, 2, 3]]);
  expect(z).toEqual(expected);
});

test('repeat matrix2', function () {
  let x = Tensor.create([
    [1, 2, 3],
    [2, 3, 4]
  ]);
  let z = TensorMath.repeat(x, 3, 0);
  // console.log(z.toString());
  let expected = Tensor.create([
    [1, 2, 3],
    [1, 2, 3],
    [1, 2, 3],
    [2, 3, 4],
    [2, 3, 4],
    [2, 3, 4]]
  );
  expect(z).toEqual(expected);
});

test('repeat matrix 3', function () {
  let x = Tensor.create([
    [1, 2, 3],
    [2, 3, 4]
  ]);
  let z = TensorMath.repeat(x, 2, 1);
  // console.log(z.toString());
  let expected = Tensor.create([
    [1, 1, 2, 2, 3, 3],
    [2, 2, 3, 3, 4, 4]
  ]);
  expect(z).toEqual(expected);
});

test('repeat 3d 0', function () {
  let x = Tensor.create(
      [[[1, 2, 3],
        [2, 3, 4]],

        [[3, 4, 5],
          [4, 5, 6]]]
  );
  let z = TensorMath.repeat(x, 2, 0);
  // console.log(z.toString());
  let expected = Tensor.create(
      [[[1, 2, 3],
        [2, 3, 4]],

        [[1, 2, 3],
          [2, 3, 4]],

        [[3, 4, 5],
          [4, 5, 6]],

        [[3, 4, 5],
          [4, 5, 6]]]
  );
  expect(z).toEqual(expected);
});

test('repeat 3d 1', function () {
  let x = Tensor.create(
      [[[1, 2, 3],
        [2, 3, 4]],

        [[3, 4, 5],
          [4, 5, 6]]]
  );
  let z = TensorMath.repeat(x, 2, 1);
  // console.log(z.toString());
  let expected = Tensor.create(
      [[[1, 2, 3],
        [1, 2, 3],
        [2, 3, 4],
        [2, 3, 4]],

        [[3, 4, 5],
          [3, 4, 5],
          [4, 5, 6],
          [4, 5, 6]]]
  );
  expect(z).toEqual(expected);
});

test('repeat 3d 2', function () {
  let x = Tensor.create(
      [[[1, 2, 3],
        [2, 3, 4]],

        [[3, 4, 5],
          [4, 5, 6]]]
  );
  let z = TensorMath.repeat(x, 2, 2);
  // console.log(z.toString());
  let expected = Tensor.create(
      [[[1, 1, 2, 2, 3, 3],
        [2, 2, 3, 3, 4, 4]],

        [[3, 3, 4, 4, 5, 5],
          [4, 4, 5, 5, 6, 6]]]
  );
  expect(z).toEqual(expected);
});


test('repeat no dim', function () {
  let x = Tensor.create([
    [1, 2, 3],
    [2, 3, 4]
  ]);
  let z = TensorMath.repeat(x, 2);
  // console.log(z.toString());
  let expected = Tensor.create([1, 1, 2, 2, 3, 3, 2, 2, 3, 3, 4, 4]);
  expect(z).toEqual(expected);
});