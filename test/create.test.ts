import TensorFactory from "../src/TensorFactory";

// test('ones', function () {
//   let tensor = TensorFactory.ones([2, 3]);
//   let expected = TensorFactory.create([[1, 1, 1], [1, 1, 1]]);
//   expect(tensor).toEqual(expected);
// });

test('zeros', function () {
  let tensor = TensorFactory.zeros([2, 3]);
  let expected = TensorFactory.create([[0, 0, 0], [0, 0, 0]]);
  expect(tensor).toEqual(expected);
});

// test('rand', function () {
//   let tensor = TensorFactory.rand([1, 1]);
//   // console.log(tensor);
// });

test('create', function () {
  let array = [1, 2, 3, 4];
  let tensor = TensorFactory.create(array);
  console.log(tensor);
  let array2 = [[1, 2], [3, 4]];
  let tensor2 = TensorFactory.create(array2);
  console.log(tensor2);
  let array3 = [[[1, 2], [3, 4], [3, 4]], [[2, 3], [4, 5], [1, 0]]];
  let tensor3 = TensorFactory.create(array3);
  console.log(tensor3);
});

// test('linspace', function () {
//   let tensor = TensorFactory.linspace(0, 5.5, 12);
//   let expected = TensorFactory.create([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5]);
//   expect(tensor).toEqual(expected);
// });

test('arange', function () {
  let tensor = TensorFactory.arange(5);
  let expected = TensorFactory.create([0, 1, 2, 3, 4]);
  expect(tensor).toEqual(expected);
});