import Shape from "../../Shape";
import Tensor from "../../Tensor";
import ShapeUtils from "../../utils/ShapeUtils";
import Operation from "../Operation"

export default class RepeatOp extends Operation {

  private _dimension: number;
  private _repeat: number;

  get dimension() {
    return this._dimension;
  }

  get isSpecial() {
    return true;
  }

  get repeat() {
    return this._repeat;
  }

  constructor(input: Tensor, result: Tensor, repeat: number, dimension: number) {
    super(input, null, result);
    this._repeat = repeat;
    this._dimension = dimension;
  }

  exec() {
    if (this.dimension === -1) {
      this.execVector();
      return;
    }

    switch (this.input.rank) {
      case 1:
        this.execVector();
        break;
      default:
        this.execGeneral();
        break;
    }
  }

  execGeneral() {
    let shape = this.input.shape;
    let length = ShapeUtils.getSlices(shape, this.dimension);

    let tempShape = this.input.shape.slice();
    tempShape[this.dimension] = 1;
    let tempShapeObj = new Shape(tempShape);

    let size = new Array(shape.length).fill(1);
    size[this.dimension] = -1;

    for (let i = 0; i < length; i++) {
      let indices = tempShapeObj.getIndices(i);
      let inputSlice = this.input.slice(indices, size);
      let resultSlice = this.result.slice(indices, size);

      let idx = 0;
      for (let j = 0; j < inputSlice.length; j++) {
        let val = inputSlice.get(j);
        for (let p = 0; p < this.repeat; p++) {
          resultSlice.set(idx, val);
          idx++;
        }
      }
    }
  }

  execVector() {
    let length = this.input.length;
    let idx = 0;
    for (let i = 0; i < length; i++) {
      let val = this.input.get(i);
      for (let p = 0; p < this.repeat; p++) {
        this.result.set(idx, val);
        idx++;
      }
    }
  }

}