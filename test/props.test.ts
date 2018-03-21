import {Tensor} from '../src/index';

test('rank=0', function() {
  let x = Tensor.create(5);
  expect(x.rank).toBe(0);
  expect(x.isScalar).toBe(true);
});

test('rank=1', function() {
  let x = Tensor.create([1, 2, 3]);
  expect(x.rank).toBe(1);
  expect(x.isVector).toBe(true);
});

test('rank=2', function() {
  let x = Tensor.create([[1, 2, 3], [4, 5, 6]]);
  expect(x.rank).toBe(2);
  expect(x.isMatrix).toBe(true);
});

test('rank=3', function() {
  let x = Tensor.create([[[1, 2, 3], [4, 5, 6]]]);
  expect(x.rank).toBe(3);
  expect(x.isMatrix).toBe(false);
});