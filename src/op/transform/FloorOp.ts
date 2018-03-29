import TransformOp from "./TransformOp";

export default class FloorOp extends TransformOp {

  body(a: number, b?: number): number {
    return Math.floor(a);
  }

}