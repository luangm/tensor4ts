/**
 * This interface is a common one for all typed arrays
 */
export default interface TensorBuffer {
  readonly BYTES_PER_ELEMENT: number;
  readonly byteLength: number;
  readonly byteOffset: number;
  readonly length: number;

  copyWithin(target: number, start: number, end?: number): this;

  fill(value: number, start?: number, end?: number): this;

  [index: number]: number;
}