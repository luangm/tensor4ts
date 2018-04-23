import PairwiseExecutor2 from "../src/executor/PairwiseExecutor2";
import AddOp from "../src/op/pairwise/AddOp";
import PairwiseOp from "../src/op/pairwise/PairwiseOp";
import Shape from "../src/Shape";
import FloatTensor from "../src/tensor/FloatTensor";

const ROW = 2048;
const COL = 2048;
const BATCH = 100;

test("base 2d", function () {
  let x = new Array(ROW * COL);
  let y = new Array(ROW * COL);
  let z = new Array(ROW * COL);

  for (let i = 0; i < ROW; i++) {
    for (let j = 0; j < COL; j++) {
      x[i * COL + j] = i * COL + j;
      y[i * COL + j] = i * COL + j;
      z[i * COL + j] = 0;
    }
  }

  let now = new Date().getTime();

  for (let n = 0; n < BATCH; n++) {
    for (let i = 0; i < ROW; i++) {
      for (let j = 0; j < COL; j++) {
        let index = i * COL + j;
        z[index] = x[index] + y[index];
      }
    }
  }

  let then = new Date().getTime();

  console.log("Finished: ", then - now);
});

test("base 1d", function () {
  let x = new Array(ROW * COL);
  let y = new Array(ROW * COL);
  let z = new Array(ROW * COL);

  for (let i = 0; i < ROW * COL; i++) {
    x[i] = i;
    y[i] = i;
    z[i] = 0;
  }

  let now = new Date().getTime();

  for (let n = 0; n < BATCH; n++) {
    for (let i = 0; i < ROW * COL; i++) {
      z[i] = x[i] + y[i];
    }
  }

  let then = new Date().getTime();

  console.log("Finished: ", then - now);
});

test("Float32Array", function () {
  let xArray = new Float32Array(ROW * COL);
  let yArray = new Float32Array(ROW * COL);
  let zArray = new Float32Array(ROW * COL);

  for (let i = 0; i < ROW; i++) {
    for (let j = 0; j < COL; j++) {
      xArray[i * COL + j] = i * COL + j;
      yArray[i * COL + j] = i * COL + j;
      zArray[i * COL + j] = 0;
    }
  }

  let now = new Date().getTime();

  for (let n = 0; n < BATCH; n++) {
    let index = 0;
    for (let i = 0; i < ROW; i++) {
      for (let j = 0; j < COL; j++) {
        // let index = i * COL + j;
        zArray[index] = xArray[index] + yArray[index];
        index++;
      }
    }
  }

  let then = new Date().getTime();

  console.log("Finished: ", then - now);
});

test("Loop Unroll", function () {
  let xArray = new Float32Array(ROW * COL);
  let yArray = new Float32Array(ROW * COL);
  let zArray = new Float32Array(ROW * COL);

  for (let i = 0; i < ROW; i++) {
    for (let j = 0; j < COL; j++) {
      xArray[i * COL + j] = i * COL + j;
      yArray[i * COL + j] = i * COL + j;
      zArray[i * COL + j] = 0;
    }
  }

  let xShape = new Shape([ROW, COL]);
  let yShape = new Shape([ROW, COL]);
  let zShape = new Shape([ROW, COL]);
  let x = new FloatTensor(xArray, xShape);
  let y = new FloatTensor(yArray, yShape);
  let z = new FloatTensor(zArray, zShape);

  let now = new Date().getTime();

  for (let n = 0; n < BATCH; n++) {
    let index = 0;
    for (let i = 0; i < ROW; i++) {
      for (let j = 0; j < COL; j += 8) {
        zArray[index] = xArray[index] + yArray[index];
        index++;
        zArray[index] = xArray[index] + yArray[index];
        index++;
        zArray[index] = xArray[index] + yArray[index];
        index++;
        zArray[index] = xArray[index] + yArray[index];
        index++;
        zArray[index] = xArray[index] + yArray[index];
        index++;
        zArray[index] = xArray[index] + yArray[index];
        index++;
        zArray[index] = xArray[index] + yArray[index];
        index++;
        zArray[index] = xArray[index] + yArray[index];
        index++;
      }
    }
  }

  let then = new Date().getTime();

  console.log("Finished: ", then - now);
});

test("Tensor", function () {
  let xArray = new Float32Array(ROW * COL);
  let yArray = new Float32Array(ROW * COL);
  let zArray = new Float32Array(ROW * COL);

  for (let i = 0; i < ROW; i++) {
    for (let j = 0; j < COL; j++) {
      xArray[i * COL + j] = i * COL + j;
      yArray[i * COL + j] = i * COL + j;
      zArray[i * COL + j] = 0;
    }
  }

  let xShape = new Shape([ROW, COL]);
  let yShape = new Shape([ROW, COL]);
  let zShape = new Shape([ROW, COL]);
  let x = new FloatTensor(xArray, xShape);
  let y = new FloatTensor(yArray, yShape);
  let z = new FloatTensor(zArray, zShape);

  let now = new Date().getTime();

  let op = new AddOp(x, y, z);

  for (let n = 0; n < BATCH; n++) {
    exec2Matrix(op);
  }

  let then = new Date().getTime();

  console.log("Finished: ", then - now, sum);
});


test("Tensor With Executor", function () {
  let xArray = new Float32Array(ROW * COL);
  let yArray = new Float32Array(ROW * COL);
  let zArray = new Float32Array(ROW * COL);

  for (let i = 0; i < ROW; i++) {
    for (let j = 0; j < COL; j++) {
      xArray[i * COL + j] = i * COL + j;
      yArray[i * COL + j] = i * COL + j;
      zArray[i * COL + j] = 0;
    }
  }

  let xShape = new Shape([ROW, COL]);
  let yShape = new Shape([ROW, COL]);
  let zShape = new Shape([ROW, COL]);
  let x = new FloatTensor(xArray, xShape);
  let y = new FloatTensor(yArray, yShape);
  let z = new FloatTensor(zArray, zShape);

  let now = new Date().getTime();

  let op = new AddOp(x, y, z);

  for (let n = 0; n < BATCH; n++) {
    PairwiseExecutor2.exec(op);
  }

  let then = new Date().getTime();

  console.log("Finished: ", then - now, sum);
});

let sum = 0;

function exec2Matrix(op: PairwiseOp): void {

  let result = op.z.data;
  let shape = op.z.shape;

  let input = op.x.data;
  let other = op.x.data;

  let inputS0 = op.x.strides[0];
  let inputS1 = op.x.strides[1];
  let otherS0 = op.y.strides[0];
  let otherS1 = op.y.strides[1];
  let resultS0 = op.z.strides[0];
  let resultS1 = op.z.strides[1];
  let s0 = shape[0];
  let s1 = shape[1];

  let leftPtr = 0;
  let rightPtr = 0;
  let resultPtr = 0;

  let inputD0 = (inputS0 - inputS1 * s1) | 0;
  let otherD0 = (otherS0 - otherS1 * s1) | 0;
  let resultD0 = (resultS0 - resultS1 * s1) | 0;

  let inputD1 = inputS1 | 0;
  let otherD1 = otherS1 | 0;
  let resultD1 = resultS1 | 0;

  let now = new Date().getTime();

  for (let i = 0; i < s0; i++) {
    for (let j = 0; j < s1; j++) {
      let a = input[leftPtr];
      let b = other[rightPtr];
      result[resultPtr] = op.body(a, b);
      leftPtr += inputD1;
      rightPtr += otherD1;
      resultPtr += resultD1;
    }

    leftPtr += inputD0;
    rightPtr += otherD0;
    resultPtr += resultD0;
  }

  let then = new Date().getTime();
  sum += then - now;

  // console.log("Finished: ", then - now);
}