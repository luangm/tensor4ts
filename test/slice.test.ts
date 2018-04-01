import {Tensor} from '../src/index';

test('slice 1', function () {
  let x = Tensor.create([
    [[1, 1, 1],
      [2, 2, 2]],

    [[3, 3, 3],
      [4, 4, 4]],

    [[5, 5, 5],
      [6, 6, 6]]
  ]);

  let b = Tensor.create([1, 2, 3]);
  console.log(b.slice([1], [2]).toString());

  let y1 = x.slice([1, 0, 0], [2, 1, 3]);
  console.log(y1.toString());

  let y2 = x.slice([1, 1, 1]);
  console.log(y2.toString());
});

test('slice get', function () {
  let x = Tensor.create([
    [1, 2, 3],
    [4, 5, 6]
  ]);

  let a = x.slice([0, 0], [2, 2]);
  console.log(a.toString());

  expect(a.get([0, 0])).toEqual(1);
  expect(a.get([0, 1])).toEqual(2);
  expect(a.get([1, 0])).toEqual(4);
  expect(a.get([1, 1])).toEqual(5);

  expect(a.get(0)).toEqual(1);
  expect(a.get(1)).toEqual(2);
  expect(a.get(2)).toEqual(4);
  expect(a.get(3)).toEqual(5);
});

test('slice get 2', function () {
  let x = Tensor.create([
    [1, 2, 3],
    [4, 5, 6]
  ]);

  let a = x.slice([0, 1], [2, 3]);
  console.log(a.toString());

  expect(a.get([0, 0])).toEqual(2);
  expect(a.get([0, 1])).toEqual(3);
  expect(a.get([1, 0])).toEqual(5);
  expect(a.get([1, 1])).toEqual(6);

  expect(a.get(0)).toEqual(2);
  expect(a.get(1)).toEqual(3);
  expect(a.get(2)).toEqual(5);
  expect(a.get(3)).toEqual(6);
});

test('slice get -1', function () {
  let x = Tensor.create([
    [1, 2, 3],
    [4, 5, 6]
  ]);

  let a = x.slice([0, 1], [-1, -1]);
  console.log(a.toString());

  expect(a.get([0, 0])).toEqual(2);
  expect(a.get([0, 1])).toEqual(3);
  expect(a.get([1, 0])).toEqual(5);
  expect(a.get([1, 1])).toEqual(6);

  expect(a.get(0)).toEqual(2);
  expect(a.get(1)).toEqual(3);
  expect(a.get(2)).toEqual(5);
  expect(a.get(3)).toEqual(6);
});

test('slice set 2', function () {
  let x = Tensor.create([
    [1, 2, 3],
    [4, 5, 6]
  ]);

  let a = x.slice([0, 1], [2, 3]);
  console.log(a.toString());

  a.set(0, 5);
  a.set(1, 6);
  a.set(2, 7);
  a.set(3, 8);

  console.log(a.toString());
  console.log(x.toString());
  // expect(a.get([0, 0])).toEqual(2);
  // expect(a.get([0, 1])).toEqual(3);
  // expect(a.get([1, 0])).toEqual(5);
  // expect(a.get([1, 1])).toEqual(6);
  //
  // expect(a.get(0)).toEqual(2);
  // expect(a.get(1)).toEqual(3);
  // expect(a.get(2)).toEqual(5);
  // expect(a.get(3)).toEqual(6);
});
