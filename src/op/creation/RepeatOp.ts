import Tensor from "../../Tensor";
import ShapeUtils from "../../utils/ShapeUtils";
import Operation from "../Operation";

export default class RepeatOp extends Operation {

  private readonly _base: Tensor;
  private readonly _dimension: number;
  private readonly _repeat: number;
  private readonly _result: Tensor;

  get base() {
    return this._base;
  }

  get dimension() {
    return this._dimension;
  }

  get isSpecial() {
    return true;
  }

  get repeat() {
    return this._repeat;
  }

  get result() {
    return this._result;
  }

  constructor(base: Tensor, result: Tensor, repeat: number, dimension: number) {
    super([base], [result]);
    this._base = base;
    this._result = result;
    this._repeat = repeat;
    this._dimension = dimension;
  }

  exec() {

  }
  //
  // exec() {
  //   if (this.dimension === -1) {
  //     this.execVector();
  //     return;
  //   }
  //
  //   switch (this.base.rank) {
  //     case 1:
  //       this.execVector();
  //       break;
  //     default:
  //       this.execGeneral();
  //       break;
  //   }
  // }
  //
  // execGeneral() {
  //   let shape = this.base.shape;
  //   let length = ShapeUtils.getSlices(shape, this.dimension);
  //
  //   let tempShape = this.base.shape.slice();
  //   tempShape[this.dimension] = 1;
  //   let tempShapeObj = new Shape(tempShape);
  //
  //   let size = new Array(shape.length).fill(1);
  //   size[this.dimension] = -1;
  //
  //   for (let i = 0; i < length; i++) {
  //     let indices = tempShapeObj.getIndices(i);
  //     let inputSlice = this.base.slice(indices, size);
  //     let resultSlice = this.result.slice(indices, size);
  //
  //     let idx = 0;
  //     for (let j = 0; j < inputSlice.length; j++) {
  //       let val = inputSlice.get(j);
  //       for (let p = 0; p < this.repeat; p++) {
  //         resultSlice.set(idx, val);
  //         idx++;
  //       }
  //     }
  //   }
  // }
  //
  // execVector() {
  //   let length = this.base.length;
  //   let idx = 0;
  //   for (let i = 0; i < length; i++) {
  //     let val = this.base.get(i);
  //     for (let p = 0; p < this.repeat; p++) {
  //       this.result.set(idx, val);
  //       idx++;
  //     }
  //   }
  // }

}