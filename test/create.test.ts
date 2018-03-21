import {Tensor} from '../src/index';

test('ones', function () {
  let tensor = Tensor.ones([2, 3]);
  let expected = Tensor.create([[1, 1, 1], [1, 1, 1]]);
  expect(tensor).toEqual(expected);
});

test('zeros', function () {
  let tensor = Tensor.zeros([2, 3]);
  let expected = Tensor.create([[0, 0, 0], [0, 0, 0]]);
  expect(tensor).toEqual(expected);
});

test('rand', function () {
  let tensor = Tensor.rand([1, 1]);
  // console.log(tensor);
});

test('create', function () {
  let array = [1, 2, 3, 4];
  let tensor = Tensor.create(array);
  let array2 = [[1, 2], [3, 4]];
  let tensor2 = Tensor.create(array2);
  let array3 = [[[1, 2], [3, 4], [3, 4]], [[2, 3], [4, 5], [1, 0]]];
  let tensor3 = Tensor.create(array3);
});

test('linspace', function () {
  let tensor = Tensor.linspace(0, 5.5, 12);
  let expected = Tensor.create([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5]);
  expect(tensor).toEqual(expected);
});

test('arange', function () {
  let tensor = Tensor.arange(5);
  let expected = Tensor.create([0, 1, 2, 3, 4]);
  expect(tensor).toEqual(expected);
});