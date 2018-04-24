import ShapeUtils from "../src/utils/ShapeUtils";

test("broadcast shape", function () {
  let a = [2, 1];
  let b = [2, 1, 3];

  let result = ShapeUtils.broadcastShapes(a, b);
  expect(result).toEqual([2, 2, 3]);
});

test("broadcast shape 2", function () {
  let a = [1];
  let b = [2, 1, 3];

  let result = ShapeUtils.broadcastShapes(a, b);
  expect(result).toEqual([2, 1, 3]);
});

test("broadcast shape scalar", function () {
  let a: number[] = [];
  let b = [2, 1, 3];

  let result = ShapeUtils.broadcastShapes(a, b);
  expect(result).toEqual([2, 1, 3]);
});