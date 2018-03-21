import TransformOp from "./TransformOp";

export default class RandomOp extends TransformOp {

  body(a: number, b?: number): number {
    return Math.random();
  }

}