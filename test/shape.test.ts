import Shape from "../src/Shape";

test('getOffset', function () {
  let x = new Shape([2, 3]);

  expect(x.getOffset([0, 0])).toEqual(0);
  expect(x.getOffset([0, 1])).toEqual(1);
  expect(x.getOffset([0, 2])).toEqual(2);
  expect(x.getOffset([1, 0])).toEqual(3);
  expect(x.getOffset([1, 1])).toEqual(4);
  expect(x.getOffset([1, 2])).toEqual(5);
});

test('getIndecies', function () {
  let x = new Shape([2, 3]);

  expect(x.getIndices(0)).toEqual([0, 0]);
  expect(x.getIndices(1)).toEqual([0, 1]);
  expect(x.getIndices(2)).toEqual([0, 2]);
  expect(x.getIndices(3)).toEqual([1, 0]);
  expect(x.getIndices(4)).toEqual([1, 1]);
  expect(x.getIndices(5)).toEqual([1, 2]);
});